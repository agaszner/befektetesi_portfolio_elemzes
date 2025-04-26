import os
import requests
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from influxdb_client import Point, WritePrecision
from .. import config
from .. import client
from .influx_db_queries import InfluxDBQueries
from .upload_to_influx import UploadToInflux

class CAPMAnalysis:
    """
    Class for performing CAPM analysis on cryptocurrency data.
    """

    MAJOR_CRYPTOS = ['ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'BTCUSDT']

    @classmethod
    def calculate_market_average_return(cls, start="2023-01-01T00:00:00Z",
                                        stop="2025-01-01T00:00:00Z", group_time='5m'):
        """
        Calculates the average return of the 5 major cryptocurrencies to use as market return.

        Args:
            start (str): Start time in ISO format (default: "2023-01-01T00:00:00Z")
            stop (str): Stop time in ISO format (default: "2025-01-01T00:00:00Z")
            group_time (str): Time interval for grouping data (default: '5m')

        Returns:
            pandas.DataFrame: DataFrame with the average market return
        """
        all_crypto_data = {}
        common_index = None

        for crypto in cls.MAJOR_CRYPTOS:
            crypto_data = InfluxDBQueries.get_data_from_influx(
                groupby_time=group_time,
                pair=crypto,
                start=start,
                stop=stop
            )

            if crypto_data.empty:
                print(f"No data available for {crypto}")
                continue

            crypto_data['return'] = crypto_data['close'].pct_change()

            crypto_data = crypto_data.dropna()

            all_crypto_data[crypto] = crypto_data

            if common_index is None:
                common_index = crypto_data.index
            else:
                common_index = common_index.intersection(crypto_data.index)

        if not all_crypto_data:
            print("No data available for any cryptocurrency")
            return pd.DataFrame()

        if len(common_index) == 0:
            print("No common time periods found across cryptocurrencies")
            return pd.DataFrame()

        market_df = pd.DataFrame(index=common_index)

        for crypto, data in all_crypto_data.items():
            market_df[crypto] = data.loc[common_index, 'return']

        market_df['market_return'] = market_df[cls.MAJOR_CRYPTOS].mean(axis=1)

        return market_df

    @classmethod
    def calculate_bitcoin_capm_with_market_average(cls, start="2023-01-01T00:00:00Z", 
                                                 stop="2025-01-01T00:00:00Z", groupby_time='5m'):
        """
        Calculates CAPM returns for Bitcoin using the average of 5 major cryptocurrencies as market return.

        Args:
            start (str): Start time in ISO format (default: "2023-01-01T00:00:00Z")
            stop (str): Stop time in ISO format (default: "2025-01-01T00:00:00Z")
            groupby_time (str): Time interval for grouping data (default: '5m')

        Returns:
            dict: Dictionary containing CAPM metrics for Bitcoin
        """

        market_df = cls.calculate_market_average_return(start, stop, groupby_time)
        if market_df.empty:
            print("Could not calculate market average return")
            return {}

        btc_data = InfluxDBQueries.get_data_from_influx(
            groupby_time=groupby_time,
            pair='BTCUSDT',
            start=start,
            stop=stop
        )

        if btc_data.empty:
            print("No data available for Bitcoin")
            return {}

        btc_data['return'] = btc_data['close'].pct_change()

        btc_data = btc_data.dropna()

        merged_data = pd.merge(
            market_df[['market_return']], 
            btc_data[['return', 'close']],
            left_index=True, 
            right_index=True,
            how='inner'
        )

        if merged_data.empty:
            print("No overlapping data for Bitcoin and market average")
            return {}

        risk_free_rate = 0.03

        if groupby_time == '5m':
            periods_per_year = 252 * 24 * 12
        elif groupby_time == '1h':
            periods_per_year = 252 * 24
        elif groupby_time == '1d':
            periods_per_year = 252
        else:
            periods_per_year = 252  # Default to daily

        risk_free_rate_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

        window_size = 288

        results = pd.DataFrame(index=merged_data.index)
        results['price'] = merged_data['close']

        rolling_cov = merged_data['return'].rolling(window=window_size).cov(merged_data['market_return'])
        rolling_var = merged_data['market_return'].rolling(window=window_size).var()
        results['beta'] = rolling_cov / rolling_var

        results['market_return'] = merged_data['market_return'].rolling(window=window_size).mean()

        results['expected_return'] = risk_free_rate_per_period + results['beta'] * (
                    results['market_return'] - risk_free_rate_per_period)

        results['actual_return'] = merged_data['return'].rolling(window=window_size).mean()

        results['alpha'] = results['actual_return'] - results['expected_return']

        results['risk_free_rate'] = risk_free_rate_per_period

        results = results.dropna()
        return results
