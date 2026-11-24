"""Central place for file paths and settings.

Every path can be overridden with an environment variable, so the same code
works on your laptop, inside Docker, inside Airflow and inside Kubernetes.
"""

import os
from pathlib import Path

# Project root = the folder that contains src/
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT_DIR / "models"))

RAW_DATA_PATH = DATA_DIR / "raw" / "btc_usd.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "btc_features.csv"

MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TICKER = os.getenv("TICKER", "BTC-USD")
FORECAST_HORIZON = int(os.getenv("FORECAST_HORIZON", "7"))

# A newly trained model is only "registered" (saved for the API) if its
# MAPE on the last 7 days is below this value (percent).
MAPE_THRESHOLD = float(os.getenv("MAPE_THRESHOLD", "15"))
