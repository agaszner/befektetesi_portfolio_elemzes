import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

env_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=env_path)

INFLUXDB_URL = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN', '')
INFLUXDB_ORG = os.getenv('INFLUXDB_ORG', 'BME')
INFLUXDB_BUCKET = os.getenv('INFLUXDB_BUCKET', 'Crypto') 