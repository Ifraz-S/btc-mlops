"""Step 2 of the pipeline: clean the raw data and build features.

The model does not see raw prices. It sees "features" that describe recent
price behaviour, all computed from the Close price only:

  * ret_lag_1 .. ret_lag_7   daily % returns for each of the last 7 days
  * ma_7_ratio, ma_30_ratio  today's price divided by its 7 / 30 day average
  * volatility_7/30          standard deviation of daily returns
  * momentum_7               % change over the last 7 days

Using only Close means the /predict endpoint can accept a plain list of
closing prices and build exactly the same features.

Targets: target_1 .. target_7 = % change from today's close to the close
1..7 days ahead. Predicting % changes (not dollar prices) is much easier for
a model because Bitcoin's price level changes a lot over the years.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

N_LAGS = 7
MIN_HISTORY = 31  # need 30 days for the 30-day features, +1 for returns


def feature_columns() -> list:
    cols = [f"ret_lag_{i}" for i in range(1, N_LAGS + 1)]
    cols += ["ma_7_ratio", "ma_30_ratio", "volatility_7", "volatility_30", "momentum_7"]
    return cols


def target_columns(horizon: int = config.FORECAST_HORIZON) -> list:
    return [f"target_{h}" for h in range(1, horizon + 1)]


def build_features(close: pd.Series) -> pd.DataFrame:
    """Build the feature table from a series of closing prices."""
    close = pd.Series(close, dtype="float64").reset_index(drop=True)
    ret = close.pct_change()
    feats = pd.DataFrame(index=close.index)
    for i in range(1, N_LAGS + 1):
        feats[f"ret_lag_{i}"] = ret.shift(i - 1)
    feats["ma_7_ratio"] = close / close.rolling(7).mean()
    feats["ma_30_ratio"] = close / close.rolling(30).mean()
    feats["volatility_7"] = ret.rolling(7).std()
    feats["volatility_30"] = ret.rolling(30).std()
    feats["momentum_7"] = close / close.shift(7) - 1
    return feats


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates("Date").sort_values("Date")
    df = df[df["Close"] > 0]
    df["Close"] = df["Close"].ffill()
    return df.reset_index(drop=True)


def make_dataset(df: pd.DataFrame, horizon: int = config.FORECAST_HORIZON) -> pd.DataFrame:
    """Full table: Date, Close, features, and future targets (NaN at the end)."""
    df = clean_raw(df)
    out = pd.concat([df[["Date", "Close"]], build_features(df["Close"])], axis=1)
    for h in range(1, horizon + 1):
        out[f"target_{h}"] = df["Close"].shift(-h) / df["Close"] - 1
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=feature_columns()).reset_index(drop=True)


def run_preprocessing(
    raw_path: Path = config.RAW_DATA_PATH, out_path: Path = config.PROCESSED_DATA_PATH
) -> Path:
    df = pd.read_csv(raw_path)
    dataset = make_dataset(df)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out_path, index=False)
    print(f"Saved {len(dataset)} feature rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    run_preprocessing()
