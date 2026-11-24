"""Kubeflow Pipeline (KFP v2 SDK) for BTC forecasting.

Three components, each runs in its own container on Kubernetes:

    ingest_data  ->  build_features  ->  train_model

Data moves between steps as KFP artifacts (Dataset, Model, Metrics), which
Kubeflow stores and tracks, so every run is reproducible and comparable.

Compile to a pipeline YAML (no Kubeflow cluster needed for this step):
    python kubeflow/btc_kfp_pipeline.py
This writes kubeflow/btc_kfp_pipeline.yaml, which can be uploaded in the
Kubeflow Pipelines UI ("Upload pipeline") and scheduled as a recurring run.

Note: KFP components must be self-contained (they run in a fresh container),
so imports happen inside each function.
"""

from pathlib import Path

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

BASE_IMAGE = "python:3.10-slim"
ML_PACKAGES = ["pandas==2.2.3", "numpy==1.26.4", "scikit-learn==1.5.2", "joblib==1.4.2"]


@dsl.component(base_image=BASE_IMAGE, packages_to_install=ML_PACKAGES + ["yfinance>=0.2.54"])
def ingest_data(ticker: str, start_date: str, raw_data: Output[Dataset]):
    """Download daily OHLCV prices from Yahoo Finance."""
    import pandas as pd
    import yfinance as yf

    df = yf.download(ticker, start=start_date, interval="1d", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna(subset=["Close"]).sort_values("Date")
    df.to_csv(raw_data.path, index=False)
    raw_data.metadata["rows"] = len(df)
    raw_data.metadata["last_date"] = str(df["Date"].iloc[-1])[:10]


@dsl.component(base_image=BASE_IMAGE, packages_to_install=ML_PACKAGES)
def build_features(raw_data: Input[Dataset], horizon: int, features: Output[Dataset]):
    """Compute returns, moving-average ratios, volatility and 1..N day targets."""
    import numpy as np
    import pandas as pd

    df = pd.read_csv(raw_data.path, parse_dates=["Date"]).drop_duplicates("Date")
    close = df["Close"].astype(float).reset_index(drop=True)
    ret = close.pct_change()
    out = df[["Date", "Close"]].reset_index(drop=True)
    for i in range(1, 8):
        out[f"ret_lag_{i}"] = ret.shift(i - 1)
    out["ma_7_ratio"] = close / close.rolling(7).mean()
    out["ma_30_ratio"] = close / close.rolling(30).mean()
    out["volatility_7"] = ret.rolling(7).std()
    out["volatility_30"] = ret.rolling(30).std()
    out["momentum_7"] = close / close.shift(7) - 1
    for h in range(1, horizon + 1):
        out[f"target_{h}"] = close.shift(-h) / close - 1
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["volatility_30", "ret_lag_7"])
    out.to_csv(features.path, index=False)
    features.metadata["rows"] = len(out)


@dsl.component(base_image=BASE_IMAGE, packages_to_install=ML_PACKAGES)
def train_model(
    features: Input[Dataset],
    horizon: int,
    model: Output[Model],
    metrics: Output[Metrics],
):
    """Backtest on the last N days, then train the final model and save it."""
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    df = pd.read_csv(features.path, parse_dates=["Date"])
    fcols = [f"ret_lag_{i}" for i in range(1, 8)] + [
        "ma_7_ratio",
        "ma_30_ratio",
        "volatility_7",
        "volatility_30",
        "momentum_7",
    ]
    tcols = [f"target_{h}" for h in range(1, horizon + 1)]
    labelled = df.dropna(subset=tcols).reset_index(drop=True)

    def make():
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_depth=4, l2_regularization=1.0
            )
        )

    origin = len(labelled) - 1
    train = labelled.iloc[: origin - horizon + 1]
    m = make().fit(train[fcols], train[tcols])
    row = labelled.iloc[[origin]]
    base = float(row["Close"].iloc[0])
    pred = base * (1 + m.predict(row[fcols])[0])
    actual = base * (1 + row[tcols].to_numpy()[0])
    err = pred - actual

    metrics.log_metric("mae", float(np.mean(np.abs(err))))
    metrics.log_metric("rmse", float(np.sqrt(np.mean(err**2))))
    metrics.log_metric("mape", float(np.mean(np.abs(err / actual)) * 100))

    final = make().fit(labelled[fcols], labelled[tcols])
    joblib.dump(
        {
            "model": final,
            "feature_columns": fcols,
            "horizon": horizon,
            "last_date": str(df["Date"].iloc[-1])[:10],
            "recent_closes": df["Close"].tail(60).round(2).tolist(),
        },
        model.path,
    )
    model.metadata["framework"] = "scikit-learn"


@dsl.pipeline(
    name="btc-forecast-training-pipeline",
    description="Ingest BTC-USD prices, engineer features, train a 7-day forecaster",
)
def btc_forecast_pipeline(
    ticker: str = "BTC-USD", start_date: str = "2018-01-01", horizon: int = 7
):
    ingest_task = ingest_data(ticker=ticker, start_date=start_date)
    ingest_task.set_display_name("Ingest BTC data")

    feature_task = build_features(raw_data=ingest_task.outputs["raw_data"], horizon=horizon)
    feature_task.set_display_name("Feature engineering")

    train_task = train_model(features=feature_task.outputs["features"], horizon=horizon)
    train_task.set_display_name("Train & evaluate")
    train_task.set_cpu_limit("1").set_memory_limit("1G")


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".yaml")
    compiler.Compiler().compile(btc_forecast_pipeline, package_path=str(out))
    print(f"Compiled pipeline -> {out}")
