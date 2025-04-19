import os
from pathlib import Path
from dotenv import load_dotenv

# Get the directory of this config file
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
env_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=env_path)

# InfluxDB configuration
INFLUXDB_URL = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN', '')
INFLUXDB_ORG = os.getenv('INFLUXDB_ORG', 'BME')
INFLUXDB_BUCKET = os.getenv('INFLUXDB_BUCKET', 'Crypto') 