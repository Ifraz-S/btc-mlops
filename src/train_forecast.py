"""Step 3 of the pipeline: train the 7-day forecasting model and save it.

Model: scikit-learn HistGradientBoostingRegressor (a fast gradient-boosted
trees model, the same family as XGBoost), one model per forecast day
("direct multi-step" strategy) via MultiOutputRegressor.

Evaluation: we hide the last 7 real days, train on everything before them,
forecast those 7 days, and compare with what actually happened
(MAE, RMSE, MAPE). A naive "price stays the same" baseline is reported too.

Registration: the new model is saved as models/model.joblib (the file the
API loads) only if its MAPE is below config.MAPE_THRESHOLD.

Run it directly:  python src/train_forecast.py
"""

import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.features import feature_columns, target_columns  # noqa: E402

CANDIDATE_MODEL = "candidate_model.joblib"
CANDIDATE_METRICS = "candidate_metrics.json"


def new_model() -> MultiOutputRegressor:
    base = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=42
    )
    return MultiOutputRegressor(base)


def save_model_file(obj, path) -> None:
    """Save the model bundle with Python's built-in C pickler.

    Airflow imports the 'dill' library, which changes the pure-Python pickler
    that joblib.dump uses. A model saved that way inside Airflow can only be
    loaded where dill is installed, and the API container doesn't have it.
    The C pickler (pickle.dump) is not affected by dill, and joblib.load in the
    API reads these files normally.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    err = predicted - actual
    return {
        "mae": round(float(np.mean(np.abs(err))), 2),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 2),
        "mape": round(float(np.mean(np.abs(err / actual)) * 100), 3),
    }


def train_and_evaluate(
    processed_path: Path = config.PROCESSED_DATA_PATH,
    model_dir: Path = config.MODEL_DIR,
    horizon: int = config.FORECAST_HORIZON,
    threshold: float = config.MAPE_THRESHOLD,
) -> dict:
    df = pd.read_csv(processed_path, parse_dates=["Date"])
    fcols, tcols = feature_columns(), target_columns(horizon)
    labelled = df.dropna(subset=tcols).reset_index(drop=True)
    if len(labelled) < 100:
        raise ValueError(f"Not enough data to train ({len(labelled)} labelled rows)")

    # --- Backtest on the most recent 7 days -------------------------------
    # 'origin' is the last day whose 7 future closes are all known.
    origin = len(labelled) - 1
    train_part = labelled.iloc[: origin - horizon + 1]  # targets end on/before origin
    model = new_model().fit(train_part[fcols], train_part[tcols])

    origin_row = labelled.iloc[[origin]]
    origin_close = float(origin_row["Close"].iloc[0])
    pred_returns = model.predict(origin_row[fcols])[0]
    predicted = origin_close * (1 + pred_returns)
    actual = origin_close * (1 + origin_row[tcols].to_numpy()[0])
    naive = np.full(horizon, origin_close)

    metrics = regression_metrics(actual, predicted)
    baseline = regression_metrics(actual, naive)

    # --- Final model trained on all labelled rows -------------------------
    final_model = new_model().fit(labelled[fcols], labelled[tcols])
    trained_at = datetime.now(timezone.utc)
    version = trained_at.strftime("v%Y%m%d-%H%M%S")

    bundle = {
        "model": final_model,
        "feature_columns": fcols,
        "horizon": horizon,
        "version": version,
        "trained_at": trained_at.isoformat(),
        "last_date": df["Date"].iloc[-1].strftime("%Y-%m-%d"),
        # keep recent closes so the API can forecast without the CSV
        "recent_closes": df["Close"].tail(60).round(2).tolist(),
    }
    report = {
        "model_version": version,
        "trained_at": bundle["trained_at"],
        "model_type": "HistGradientBoostingRegressor (direct multi-output)",
        "train_rows": int(len(labelled)),
        "validation_window": {
            "start": str(labelled["Date"].iloc[origin].date() + pd.Timedelta(days=1)),
            "days": horizon,
        },
        "metrics": metrics,
        "naive_baseline_metrics": baseline,
        "mape_threshold": threshold,
        "passed_threshold": metrics["mape"] < threshold,
        "validation_actual": [round(float(x), 2) for x in actual],
        "validation_predicted": [round(float(x), 2) for x in predicted],
    }

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_model_file(bundle, model_dir / CANDIDATE_MODEL)
    (model_dir / CANDIDATE_METRICS).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("model_version", "metrics", "passed_threshold")}))
    return report


def register_model(model_dir: Path = config.MODEL_DIR, force: bool = False) -> bool:
    """Promote the candidate to model.joblib if it passed the threshold.

    If no model exists yet, the candidate is always registered so the API
    has something to serve.
    """
    model_dir = Path(model_dir)
    report = json.loads((model_dir / CANDIDATE_METRICS).read_text())
    first_model = not (model_dir / "model.joblib").exists()
    if report["passed_threshold"] or force or first_model:
        # copyfile copies only the contents. shutil.copy would also try to copy
        # file permissions, which fails on Windows/Docker shared folders when
        # the existing file was created by a different user.
        shutil.copyfile(model_dir / CANDIDATE_MODEL, model_dir / "model.joblib")
        shutil.copyfile(model_dir / CANDIDATE_METRICS, model_dir / "metrics.json")
        print(f"Registered model {report['model_version']}")
        return True
    print(f"Model {report['model_version']} NOT registered (MAPE above threshold)")
    return False


if __name__ == "__main__":
    train_and_evaluate()
    register_model()