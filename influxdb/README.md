# InfluxDB Directory

This directory contains all InfluxDB-related components:

## Structure
- **commands/**: Python modules for interacting with InfluxDB (queries, uploads)
- **data/**: InfluxDB persistent data storage
- **.env**: Environment variables for InfluxDB configuration (token, bucket, etc.)
- **config.py**: Configuration module to load settings from .env file
- **client.py**: Centralized client factory for creating InfluxDB clients

## Environment Variables
The following environment variables are used:
- `INFLUXDB_URL`: The URL of the InfluxDB server (default: http://localhost:8086)
- `INFLUXDB_TOKEN`: Authentication token for InfluxDB
- `INFLUXDB_ORG`: Organization name in InfluxDB (default: BME)
- `INFLUXDB_BUCKET`: Default bucket name (default: Crypto)

## Usage

### Importing Components
```python
from influxdb import UploadToInflux, InfluxDBQueries
```

### Getting InfluxDB Clients
```python
from influxdb import client

# Get a basic InfluxDB client
db_client = client.get_client()

# Get a write API with default batch settings
write_api = client.get_write_api()

# Get a query API for executing FluxQL queries
query_api = client.get_query_api()
```

The docker-compose.yml file has been updated to use this directory structure. 