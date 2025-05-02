import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from .capm_analysis import CAPMAnalysis
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class DataProcessor():

    @classmethod
    def make_sequences(cls, df, columns_to_select, window_size=288, forecast_horizon=12, classification=False):
        """
        Creates input-output sequences for LSTM training.

        :param df: pandas DataFrame with your time series data
        :param columns_to_select: list of column names to use as features (for X)
        :param window_size: number of time steps to use for input
        :param forecast_horizon: number of future steps to predict
        :param classification: if True, y will be a binary classification target
        :return: tuple (X, y) where:
                 - X is a numpy array of shape (num_samples, window_size, num_features)
                 - y is a numpy array of shape (num_samples, forecast_horizon, 1)
                   representing either price changes or absolute prices for each 5-min interval
        """
        data_X = df[columns_to_select].values
        data_y = df['target'].values
        data_close = df['close'].values
        X, y = [], []

        for i in range(len(df) - window_size - forecast_horizon + 1):
            X.append(data_X[i:i + window_size])

            if forecast_horizon == 1:
                close_prices = data_y[i + window_size - 1:i + window_size + forecast_horizon]
            else:
                close_prices = data_close[i + window_size - 1:i + window_size + forecast_horizon]

            if forecast_horizon > 1:
                base_price = close_prices[0]
                y_values = ((close_prices[1:] - base_price) / base_price)
            else:
                y_values = close_prices[0]

            y.append(y_values)

        X_array = np.array(X)

        y_array = np.array(y)

        if classification:
            y_array = y_array.reshape(-1, 1)
        else:
            if forecast_horizon > 1:
                y_array = y_array.reshape(-1, forecast_horizon, 1)
            else:
                y_array = y_array.reshape(-1, 1, 1)

        return X_array, y_array

    @classmethod
    def add_features(cls, df, classification=False, group_time='5m'):
        """
        Input:
          df indexed by 5‑min timestamps, columns: ['open','high','low','close','volume']
        Output:
          df with new feature columns for modeling 1h‑ahead moves.
        Requires helper funcs: compute_rsi, compute_macd, compute_atr, compute_obv (as before).
        """
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)

        # Rolling statistics (mean, std) and range for windows 5,12,24
        for w in [24]:
            df[f'close_ma_{w}'] = df['close'].rolling(window=w).mean()
            df[f'close_std_{w}'] = df['close'].rolling(window=w).std()
            df[f'return_mean_{w}'] = df['return_1'].rolling(window=w).mean()
            df[f'return_std_{w}'] = df['return_1'].rolling(window=w).std()
            df[f'range_{w}'] = df['high'].rolling(w).max() - df['low'].rolling(w).min()
            df[f'vol_mean_{w}'] = df['volume'].rolling(window=w).mean()
            df[f'vol_std_{w}'] = df['volume'].rolling(window=w).std()

        # Trend features: simple moving averages (fast and slow) and their difference
        df['sma_fast'] = df['close'].rolling(window=5).mean()
        df['sma_slow'] = df['close'].rolling(window=24).mean()
        df['ma_diff'] = df['sma_fast'] - df['sma_slow']

        # Momentum indicators
        df['rsi_14'] = cls.compute_rsi(df['close'], window=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = cls.compute_macd(df['close'], fast=12, slow=26, signal=9)
        # Stochastic Oscillator (fast %K and slow %D)
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # Volatility indicators
        # True Range and ATR (14-period)&#8203;:contentReference[oaicite:3]{index=3}
        df['tr'] = np.maximum.reduce([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ])
        df['atr_14'] = df['tr'].rolling(window=14).mean()
        # Bollinger Bands (20-period, 2*std)
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_width'] = df['bb_upper'] - df['bb_lower']

        # Volume features
        # VWAP (24-period rolling)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).rolling(window=24).sum() / df['volume'].rolling(window=24).sum()

        # Liquidity / market structure
        df['obv'] = cls.compute_obv(df['close'], df['volume'])
        df['range_cur'] = df['high'] - df['low']
        df['range_pct'] = df['range_cur'] / df['close']

        capm = CAPMAnalysis.calculate_bitcoin_capm_with_market_average(
            start=df.index.min().strftime('%Y-%m-%dT%H:%M:%SZ'), stop=df.index.max().strftime('%Y-%m-%dT%H:%M:%SZ'), groupby_time=group_time)

        df = df.merge(capm, on='time', how='left')
        df.dropna(inplace=True)

        if classification:
            df['target'] = (df['close'].shift(-1) - df['close']) / df['close']
            df['target'] = np.where(df['target'] > 0, 1, 0)
            target_scaler = None
        else:
            df['target'] = (df['close'].shift(-1) - df['close']) / df['close']
            target_scaler = StandardScaler()
            df['target'] = target_scaler.fit_transform(df[['target']])

        feature_cols = [col for col in df.columns if col not in ['tr','bb_mid','bb_std', 'target']]
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

        #save scaler
        scaler_path = f'./scalers/scaler{group_time}.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        return df, scaler, feature_cols, target_scaler

    @classmethod
    def compute_rsi(cls, series, window=14):
        """
        Compute Relative Strength Index (RSI) using Wilder's smoothing.
        RSI = 100 - (100/(1 + RS)), where RS = avg_gain/avg_loss&#8203;:contentReference[oaicite:0]{index=0}.
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @classmethod
    def compute_macd(cls, series, fast=12, slow=26, signal=9):
        """
        Compute MACD line, signal line, and histogram.
        MACD = EMA_fast - EMA_slow; Signal = EMA(MACD, signal); Histogram = MACD - Signal&#8203;:contentReference[oaicite:1]{index=1}.
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - signal_line
        return macd_line, signal_line, macd_hist

    @classmethod
    def compute_obv(cls, close, volume):
        """
        Compute On-Balance Volume (OBV) cumulative.
        OBV increases by volume on up days and decreases on down days&#8203;:contentReference[oaicite:2]{index=2}.
        """
        signs = np.sign(close.diff().fillna(0))
        obv = (signs * volume).cumsum()
        return obv

    @classmethod
    def select_features_boruta(cls,
                               df,
                               feature_cols,
                               target_col='target',
                               sample_size=100000,
                               n_estimators='auto',
                               random_state=42,
                               classification=False):
        """
        Boruta feature selection, regression vagy klaszifikáció esetén.

        :param df: pandas DataFrame
        :param feature_cols: lista a feature oszlopnevekkel
        :param target_col: célváltozó oszlop neve (folyamatos vagy bináris)
        :param classification: bool, ha True → klaszifikáció (RFC), egyébként regresszió (RFR)
        :param sample_size: max minta elemszám
        :param n_estimators: 'auto' vagy int, a fák száma (ha 'auto', Boruta belsőleg kezeli)
        :param max_iter: Boruta iterációk száma
        :param random_state: seed
        :return: (selected_features, boruta_selector)
        """
        # mintavételezés
        sample_df = df.sample(n=sample_size, random_state=random_state) \
            if len(df) > sample_size else df

        X = sample_df[feature_cols].values
        y = sample_df[target_col].values.ravel()

        # modell kiválasztása
        if classification:
            estimator = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
                max_depth = 3,
                class_weight = 'balanced',
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators='200',
                random_state=random_state,
                n_jobs=-1
            )

        # Boruta selector létrehozása
        boruta_selector = BorutaPy(
            estimator=estimator,
            random_state=random_state,
            n_estimators=n_estimators,
            verbose=2,
            alpha=0.05,
            perc= 30,
            max_iter=200,
        )
        boruta_selector.fit(X, y)

        # eredmények kiírása
        print("\n------ Feature Selection Results ------\n")
        results = pd.DataFrame({
            'Feature': feature_cols,
            'Support': boruta_selector.support_,
            'Ranking': boruta_selector.ranking_
        })
        results = results.sort_values(by=['Support', 'Ranking'], ascending=[False, True])
        print(results.to_string(index=False))

        selected_features = results.loc[results['Support'], 'Feature'].tolist()
        print(f"\nSelected features ({'classification' if classification else 'regression'}):",
              selected_features)

        return selected_features, boruta_selector