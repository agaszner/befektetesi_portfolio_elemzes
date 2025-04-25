import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import isoparse
from .. import config
from .. import client

class InfluxDBQueries():

    @classmethod
    def get_data_from_influx(cls, groupby_time='1d', pair='BTCUSDT',
                             start="2022-01-01T01:00:00Z", stop="2025-01-01T01:00:00Z",
                             chunk_days=90):
        bucket = config.INFLUXDB_BUCKET

        # Parse start and stop into datetime objects
        start_dt = isoparse(start)
        stop_dt = isoparse(stop)

        # Get client and ensure it's closed after use
        influx_client = client.get_client()
        all_dfs = []

        try:
            # Get query API from the client
            query_api = influx_client.query_api()

            while start_dt < stop_dt:
                chunk_stop_dt = min(start_dt + timedelta(days=chunk_days), stop_dt)

                query = f"""
                from(bucket: "{bucket}")
                  |> range(start: {start_dt.isoformat()}, stop: {chunk_stop_dt.isoformat()})
                  |> filter(fn: (r) => r["_measurement"] == "crypto" and r["_field"] == "open" and r["pair"] == "{pair}")
                  |> aggregateWindow(every: {groupby_time}, fn: first, createEmpty: false)
                  |> yield(name: "open")

                from(bucket: "{bucket}")
                  |> range(start: {start_dt.isoformat()}, stop: {chunk_stop_dt.isoformat()})
                  |> filter(fn: (r) => r["_measurement"] == "crypto" and r["_field"] == "high" and r["pair"] == "{pair}")
                  |> aggregateWindow(every: {groupby_time}, fn: max, createEmpty: false)
                  |> yield(name: "high")

                from(bucket: "{bucket}")
                  |> range(start: {start_dt.isoformat()}, stop: {chunk_stop_dt.isoformat()})
                  |> filter(fn: (r) => r["_measurement"] == "crypto" and r["_field"] == "low" and r["pair"] == "{pair}")
                  |> aggregateWindow(every: {groupby_time}, fn: min, createEmpty: false)
                  |> yield(name: "low")

                from(bucket: "{bucket}")
                  |> range(start: {start_dt.isoformat()}, stop: {chunk_stop_dt.isoformat()})
                  |> filter(fn: (r) => r["_measurement"] == "crypto" and r["_field"] == "close" and r["pair"] == "{pair}")
                  |> aggregateWindow(every: {groupby_time}, fn: last, createEmpty: false)
                  |> yield(name: "close")

                from(bucket: "{bucket}")
                  |> range(start: {start_dt.isoformat()}, stop: {chunk_stop_dt.isoformat()})
                  |> filter(fn: (r) => r["_measurement"] == "crypto" and r["_field"] == "volume" and r["pair"] == "{pair}")
                  |> aggregateWindow(every: {groupby_time}, fn: sum, createEmpty: false)
                  |> yield(name: "volume")
                """

                result = query_api.query(query)
                records_dict = {}

                for table in result:
                    for record in table.records:
                        time = record.get_time()
                        field = record.get_field()
                        value = record.get_value()

                        if time not in records_dict:
                            records_dict[time] = {}
                        records_dict[time][field] = value

                df = pd.DataFrame.from_dict(records_dict, orient='index')
                if not df.empty:
                    df.index.name = 'time'
                    df = df.sort_index()
                    all_dfs.append(df)

                # Move to next chunk
                start_dt = chunk_stop_dt

            # Combine all chunks
            if all_dfs:
                full_df = pd.concat(all_dfs)
                full_df = full_df[~full_df.index.duplicated(keep='first')]  # Remove any potential overlaps
                return full_df.sort_index()
            else:
                print("Nincs adat a megadott időintervallumban.")
                return pd.DataFrame()
        finally:
            # Ensure client is closed even if an exception occurs
            influx_client.close()
