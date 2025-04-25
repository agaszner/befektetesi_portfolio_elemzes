# Python
from .. import config
from .. import client

class DeleteInfluxData:
    def __init__(self):
        self.influxdb_client = client.get_client()

    def delete_data(self, start_time, end_time, predicate=""):
        """
        Delete data using configuration parameters.

        Args:
            start_time (str): Start of the deletion range (ISO8601 format)
            end_time (str): End of the deletion range (ISO8601 format)
            predicate (str): Optional predicate
        """
        delete_api = self.influxdb_client.delete_api()
        delete_api.delete(
            start=start_time,
            stop=end_time,
            predicate=predicate,
            bucket=config.INFLUXDB_BUCKET,
            org=config.INFLUXDB_ORG
        )
        print(f"Data deleted from {start_time} to {end_time} in bucket {config.INFLUXDB_BUCKET}.")