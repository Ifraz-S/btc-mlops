"""Shared test setup (pytest loads this file automatically).

The tests must run in GitHub Actions without internet access to Yahoo, so we
generate a realistic-looking fake BTC price series, train a real model on
it in a temporary folder, and point the API at that folder.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import make_dataset  # noqa: E402
from src.train_forecast import register_model, train_and_evaluate  # noqa: E402


def synthetic_prices(n_days: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0.001, 0.03, n_days)))
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1e9, 5e9, n_days),
        }
    )


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("models")
    processed = tmp / "features.csv"
    make_dataset(synthetic_prices()).to_csv(processed, index=False)
    train_and_evaluate(processed_path=processed, model_dir=tmp)
    register_model(model_dir=tmp)
    return tmp


@pytest.fixture(scope="session")
def api_module(model_dir):
    spec = importlib.util.spec_from_file_location("btc_api", ROOT / "app" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_DIR = model_dir
    assert module.load_model()
    return module


@pytest.fixture()
def client(api_module):
    api_module.app.config["TESTING"] = True
    return api_module.app.test_client()
