"""
Centralized InfluxDB client creation module.
"""
from influxdb_client import InfluxDBClient, WriteOptions
from . import config

def get_client():
    """
    Returns a configured InfluxDBClient using settings from config.
    
    Returns:
        InfluxDBClient: A configured InfluxDB client
    """
    return InfluxDBClient(
        url=config.INFLUXDB_URL, 
        token=config.INFLUXDB_TOKEN, 
        org=config.INFLUXDB_ORG,
        timeout=1_000_000,
    )

def get_write_api(batch_size=5000, flush_interval=10_000):
    """
    Returns a write API with the specified batch settings.
    
    Args:
        batch_size (int): Number of points to write in a single batch
        flush_interval (int): Flush interval in milliseconds
        
    Returns:
        WriteApi: A configured write API
    """
    client = get_client()
    write_options = WriteOptions(
        write_type="batch", 
        batch_size=batch_size, 
        flush_interval=flush_interval
    )
    return client.write_api(write_options=write_options)

def get_query_api():
    """
    Returns a query API for executing FluxQL queries.
    
    Returns:
        QueryApi: A configured query API
    """
    client = get_client()
    return client.query_api() 