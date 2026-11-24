"""Unit tests for feature engineering and training."""

import json

import pandas as pd

from src.features import MIN_HISTORY, build_features, feature_columns, make_dataset
from tests.conftest import synthetic_prices


def test_features_have_no_gaps_after_warmup():
    feats = build_features(synthetic_prices(100)["Close"])
    assert list(feats.columns) == feature_columns()
    assert not feats.iloc[MIN_HISTORY - 1 :].isna().any(axis=None)


def test_dataset_targets_are_future_returns():
    df = synthetic_prices(100)
    ds = make_dataset(df)
    row = ds.iloc[0]
    later = ds[ds["Date"] == row["Date"] + pd.Timedelta(days=1)].iloc[0]
    assert abs(row["target_1"] - (later["Close"] / row["Close"] - 1)) < 1e-9


def test_training_writes_model_and_metrics(model_dir):
    assert (model_dir / "model.joblib").exists()
    report = json.loads((model_dir / "metrics.json").read_text())
    assert report["metrics"]["mape"] >= 0
    assert len(report["validation_predicted"]) == 7
