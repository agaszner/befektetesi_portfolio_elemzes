import os
import requests
import zipfile
import pandas as pd
from datetime import datetime, timedelta
from influxdb_client import Point, WritePrecision
from .. import config
from .. import client

class UploadToInflux:

    @classmethod
    def download_binance_data(cls, year, month: str, day: str, symbol='BTCUSDT', interval='1s'):
        base_url = f"https://data.binance.vision/data/spot/daily/klines/{symbol}/{interval}/"
        file_date = f"{year}-{month}-{day}"
        file_url = f"{base_url}{symbol}-{interval}-{file_date}.zip"
        data_dir = "./data"

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        try:
            response = requests.get(file_url, stream=True)
            response.raise_for_status()
            zip_path = os.path.join(data_dir, f"{symbol}-{interval}-{file_date}.zip")

            with open(zip_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {zip_path}")

            csv_path = os.path.join(data_dir, f"{symbol}-{interval}-{file_date}.csv")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                csv_file = [f for f in zip_ref.namelist() if f.endswith('.csv')][0]
                zip_ref.extract(csv_file, data_dir)
                os.rename(os.path.join(data_dir, csv_file), csv_path)
                print(f"Extracted to {csv_path}")
                os.remove(zip_path)

            cls._load_to_influxdb(csv_path)
            os.remove(csv_path)

        except Exception as e:
            print(f"Hiba {file_date} esetén: {str(e)}")

    @classmethod
    def _load_to_influxdb(cls, file_path):
        bucket = config.INFLUXDB_BUCKET
        org = config.INFLUXDB_ORG

        # Get client and write API
        influx_client = client.get_client()

        try:
            write_api = influx_client.write_api(write_options=client.WriteOptions(
                write_type="batch", 
                batch_size=5000, 
                flush_interval=10_000
            ))

            columns = [ 'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'tb_base', 'tb_quote', 'ignore']

            df = pd.read_csv(file_path, header=None, names=columns, index_col=False)
            points = []
            for _, row in df.iterrows():
                timestamp_ns = int(row['open_time'] / 1000)
                point = (Point("crypto").tag("pair", "BTCUSDT")
                        .time(timestamp_ns, WritePrecision.MS)
                        .field("open", float(row['open']))
                        .field("high", float(row['high']))
                        .field("low", float(row['low']))
                        .field("close", float(row['close']))
                        .field("volume", float(row['volume'])))
                points.append(point)

            write_api.write(bucket=bucket, org=org, record=points)
            print(f"{len(points)} adatpont feltöltve: {file_path}")

        except Exception as e:
            print(f"Hiba a {file_path} feldolgozásában: {str(e)}")
        finally:
            # Ensure client is closed even if an exception occurs
            influx_client.close()

    @classmethod
    def upload_after_last_upload(cls, pair='BTCUSDT', days=30):
        bucket = config.INFLUXDB_BUCKET

        # Get client and ensure it's closed after use
        influx_client = client.get_client()

        try:
            # Get query API from the client
            query_api = influx_client.query_api()

            query = f"""from(bucket: "{bucket}")
                          |> range(start: -{days}d)
                          |> filter(fn: (r) => r["_measurement"] == "crypto")
                          |> filter(fn: (r) => r["pair"] == "{pair}")
                          |> last()
            """

            result = query_api.query(query)
            last_measurement = None
            for table in result:
                for record in table.records:
                    last_measurement = record.get_time()
            if last_measurement is None:
                print("Nincs mérés az elmúlt 30 napban.")
                return

            start_date = last_measurement.date() + timedelta(days=1)
            end_date = datetime.now().date()

            current_date = start_date
            while current_date <= end_date:
                year = current_date.year
                month = f"{current_date.month:02d}"
                day = f"{current_date.day:02d}"
                cls.download_binance_data(year, month, day)
                current_date += timedelta(days=1)
        finally:
            # Ensure client is closed even if an exception occurs
            influx_client.close()
