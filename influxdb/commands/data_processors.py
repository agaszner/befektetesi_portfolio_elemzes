import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .capm_analysis import CAPMAnalysis


class DataProcessor():

    @classmethod
    def make_sequences(cls, df, columns_to_select, window_size=288, forecast_horizon=12, return_absolute_prices=False):
        """
        Creates input-output sequences for LSTM training.

        :param df: pandas DataFrame with your time series data
        :param columns_to_select: list of column names to use as features (for X)
        :param window_size: number of time steps to use for input
        :param forecast_horizon: number of future steps to predict
        :param return_absolute_prices: if True, returns absolute prices instead of price changes
        :return: tuple (X, y) where:
                 - X is a numpy array of shape (num_samples, window_size, num_features)
                 - y is a numpy array of shape (num_samples, forecast_horizon, 1)
                   representing either price changes or absolute prices for each 5-min interval
        """
        data_X = df[columns_to_select].values
        data_y = df['target'].values  # 'close' column for calculating changes
        X, y = [], []

        for i in range(len(df) - window_size - forecast_horizon + 1):
            X.append(data_X[i:i + window_size])

            # Get the close prices for the forecast horizon plus the last point of the window
            close_prices = data_y[i + window_size - 1:i + window_size + forecast_horizon]

            if return_absolute_prices:
                # Use absolute prices for the forecast horizon (excluding the last point of the window)
                y_values = close_prices[1:]
            else:
                y_values = close_prices[0]

            y.append(y_values)

        X_array = np.array(X)
        y_array = np.array(y).reshape(-1, forecast_horizon)

        return X_array, y_array

    @classmethod
    def add_features(cls, df):
        """
        Input:
          df indexed by 5‑min timestamps, columns: ['open','high','low','close','volume']
        Output:
          df with new feature columns for modeling 1h‑ahead moves.
        Requires helper funcs: compute_rsi, compute_macd, compute_atr, compute_obv (as before).
        """
        # 1) Price lag & returns
        scaler = StandardScaler()
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)

        # 2) Rolling stats on price
        df['rolling_mean_12'] = df['close'].rolling(12).mean()
        df['rolling_std_12']  = df['close'].rolling(12).std()

        # 3) Price range & typical price
        df['price_range']    = df['high'] - df['low']
        df['typical_price']  = (df['high'] + df['low'] + df['close']) / 3

        # 4) VWAP (20‑period rolling)
        df['vwap_20'] = (
            (df['typical_price'] * df['volume']).rolling(20).sum()
            / df['volume'].rolling(20).sum()
        )

        # 5) Volume stats
        df['vol_mean_10'] = df['volume'].rolling(10).mean()
        df['vol_std_10']  = df['volume'].rolling(10).std()

        # 6) Moving averages
        df['ma_7']  = df['close'].rolling(7).mean()
        df['ma_30'] = df['close'].rolling(30).mean()

        # 7) RSI (14)
        df['rsi_14'] = cls.compute_rsi(df['close'], window=14)

        # 8) Bollinger Bands (20,2)
        df['bb_mean_20'] = df['close'].rolling(20).mean()
        df['bb_std_20']  = df['close'].rolling(20).std()
        df['bb_hband']   = df['bb_mean_20'] + 2 * df['bb_std_20']
        df['bb_lband']   = df['bb_mean_20'] - 2 * df['bb_std_20']

        # 9) MACD & signal
        macd_line, macd_signal = cls.compute_macd(df['close'])
        df['macd']        = macd_line
        df['macd_signal'] = macd_signal

        # 10) ATR (14)
        df['atr'] = cls.compute_atr(df['high'], df['low'], df['close'], window=24)

        # 11) OBV (price‑based)
        df['obv'] = cls.compute_obv(df['close'], df['volume'])
        start = df.index.min().strftime('%Y-%m-%dT%H:%M:%SZ')
        end = df.index.max().strftime('%Y-%m-%dT%H:%M:%SZ')
        capm = CAPMAnalysis.calculate_bitcoin_capm_with_market_average(start=start, stop=end)
        df = df.merge(capm, on='time', how='left')
        df.dropna(inplace=True)
        df['target'] = (df['close'].shift(12) - df['close']) / df['close']
        df['last_direction'] = np.where(df['target'].shift(1) > 0, 1, 0)
        df['last_movement'] = df['target'].shift(1)

        #columns to scale
        columns = ['rolling_mean_12', 'rolling_std_12', 'price_range', 'typical_price',
                     'vwap_20', 'vol_mean_10', 'vol_std_10', 'ma_7', 'ma_30',
                     'rsi_14', 'bb_mean_20', 'bb_std_20', 'bb_hband', 'bb_lband',
                     'macd', 'macd_signal', 'atr', 'obv', 'alpha', 'beta', 'market_return', 'expected_return', 'actual_return', 'last_movement',
                   'last_direction']

        df[columns] = scaler.fit_transform(df[columns])
        y_scaler = StandardScaler()
        df['target'] = y_scaler.fit_transform(df[['target']])
        return df, scaler, columns, y_scaler

    @classmethod
    def compute_rsi(cls, series: pd.Series, window: int = 14) -> pd.Series:
        """
        Compute the Relative Strength Index (RSI) for a given series.
        The RSI is a momentum oscillator that measures the speed and change of price movements.

        Args:
            series (pd.Series): The input time series data
            window (int): The window size for the RSI calculation
        """
        delta = series.diff()
        delta = delta[1:]

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Use simple moving average of gains/losses
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @classmethod
    def compute_ema(cls, series: pd.Series, span: int) -> pd.Series:
        """
        Exponential moving average (EMA)
        """
        return series.ewm(span=span, adjust=False).mean()

    @classmethod
    def compute_macd(cls, series: pd.Series) -> (pd.Series, pd.Series):
        """
        MACD line and signal line
        - MACD line = EMA12 − EMA26
        - Signal line = 9‑period EMA of MACD line
        """
        ema12 = cls.compute_ema(series, span=12)
        ema26 = cls.compute_ema(series, span=26)
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        return macd_line, signal_line

    @classmethod
    def compute_atr(cls, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Average True Range (ATR)
        """
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window).mean()
        return atr

    @classmethod
    def compute_obv(cls, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume (OBV)
        """
        # +volume on up days, -volume on down days
        direction = np.sign(close.diff()).fillna(0)
        obv = (direction * volume).cumsum()
        return obv

