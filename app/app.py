"""Flask REST API that serves the 7-day Bitcoin closing price forecast.

Endpoints
---------
GET  /health   -> is the service alive and is a model loaded?
POST /predict  -> 7-day forecast (optionally from your own price history)
GET  /metrics  -> latest validation metrics of the loaded model
POST /reload   -> re-read the model file from disk (called by Airflow)

Run for development:  python app/app.py
Run for production:   gunicorn --bind 0.0.0.0:5000 app:app   (from inside app/)
"""

import json
import logging
import os
import sys
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):  # works both in the repo and inside Docker
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.features import MIN_HISTORY, build_features  # noqa: E402

MODEL_DIR = Path(os.getenv("MODEL_DIR", HERE.parent / "models"))
MODEL_STAGE = os.getenv("MODEL_STAGE", "production")
API_TOKEN = os.getenv("API_TOKEN", "")  # comes from a Kubernetes Secret
MAX_HISTORY = 5000

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("btc-api")

app = Flask(__name__)
START_TIME = time.time()
_lock = threading.Lock()
_state = {"bundle": None, "metrics": {}, "loaded_at": None}
_stats = {"predict_requests": 0, "predict_errors": 0}


class BadRequest(Exception):
    """Raised for invalid user input -> HTTP 400."""


# ---------------------------------------------------------------- model I/O
def load_model() -> bool:
    model_path = MODEL_DIR / "model.joblib"
    metrics_path = MODEL_DIR / "metrics.json"
    if not model_path.exists():
        log.warning("No model found at %s", model_path)
        return False
    bundle = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    with _lock:
        _state.update(bundle=bundle, metrics=metrics, loaded_at=time.time())
    log.info("Loaded model %s", bundle.get("version"))
    return True


def get_bundle():
    with _lock:
        return _state["bundle"]


# ---------------------------------------------------------------- helpers
def parse_payload() -> dict:
    """Return the JSON body as a dict. An empty body means 'use defaults'."""
    if not request.data:
        return {}
    body = request.get_json(silent=True)
    if body is None:
        raise BadRequest("Request body must be valid JSON")
    if not isinstance(body, dict):
        raise BadRequest('JSON body must be an object, e.g. {"forecast_days": 7}')
    return body


def validate_forecast_days(value, horizon: int) -> int:
    if value is None:
        return horizon
    if isinstance(value, bool) or not isinstance(value, int):
        raise BadRequest("forecast_days must be an integer")
    if not 1 <= value <= horizon:
        raise BadRequest(f"forecast_days must be between 1 and {horizon}")
    return value


def validate_history(value) -> list:
    if not isinstance(value, list):
        raise BadRequest("history must be a list of closing prices")
    if len(value) < MIN_HISTORY:
        raise BadRequest(f"history needs at least {MIN_HISTORY} daily closing prices")
    if len(value) > MAX_HISTORY:
        raise BadRequest(f"history can have at most {MAX_HISTORY} values")
    for v in value:
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
            raise BadRequest("history must contain only numbers (no nulls)")
        if not np.isfinite(v) or v <= 0:
            raise BadRequest("history values must be positive finite numbers")
    return [float(v) for v in value]


def validate_date(value) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise BadRequest("last_date must be in YYYY-MM-DD format")


def forecast(bundle: dict, closes: list, last_day: date, days: int) -> list:
    feats = build_features(closes)[bundle["feature_columns"]].iloc[[-1]]
    if feats.isna().any(axis=None):
        raise BadRequest("Could not compute features from history")
    returns = bundle["model"].predict(feats)[0][:days]
    last_close = closes[-1]
    return [
        {
            "date": (last_day + timedelta(days=i + 1)).isoformat(),
            "predicted_close": round(float(last_close * (1 + r)), 2),
        }
        for i, r in enumerate(returns)
    ]


# ---------------------------------------------------------------- routes
@app.get("/health")
def health():
    bundle = get_bundle()
    body = {
        "status": "ok" if bundle else "model_not_loaded",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model_version": bundle.get("version") if bundle else None,
        "model_stage": MODEL_STAGE,
        "pod": os.getenv("HOSTNAME", "local"),
    }
    return jsonify(body), (200 if bundle else 503)


@app.post("/predict")
def predict():
    _stats["predict_requests"] += 1
    bundle = get_bundle()
    if bundle is None:
        _stats["predict_errors"] += 1
        return jsonify(error="Model not loaded yet"), 503
    try:
        body = parse_payload()
        days = validate_forecast_days(body.get("forecast_days"), bundle["horizon"])
        if "history" in body:
            closes = validate_history(body["history"])
            last_day = validate_date(body["last_date"]) if "last_date" in body else date.today()
            source = "request_history"
        else:
            closes = bundle["recent_closes"]
            last_day = date.fromisoformat(bundle["last_date"])
            source = "latest_training_data"
        predictions = forecast(bundle, closes, last_day, days)
    except BadRequest as exc:
        _stats["predict_errors"] += 1
        return jsonify(error=str(exc)), 400
    except Exception:
        _stats["predict_errors"] += 1
        log.exception("Prediction failed")
        return jsonify(error="Internal error while predicting"), 500

    return jsonify(
        ticker="BTC-USD",
        model_version=bundle["version"],
        input_source=source,
        last_known_date=last_day.isoformat(),
        last_known_close=round(closes[-1], 2),
        forecast_days=days,
        predictions=predictions,
    )


@app.get("/metrics")
def metrics():
    with _lock:
        m = dict(_state["metrics"])
    if not m:
        return jsonify(error="No metrics available, model not loaded"), 503
    m["api_stats"] = dict(_stats)
    return jsonify(m)


@app.post("/reload")
def reload_model():
    if API_TOKEN and request.headers.get("X-API-Token") != API_TOKEN:
        return jsonify(error="Invalid or missing X-API-Token header"), 401
    try:
        ok = load_model()
    except Exception:
        log.exception("Reload failed")
        return jsonify(error="Failed to load model"), 500
    if not ok:
        return jsonify(error="Model file not found"), 404
    return jsonify(status="reloaded", model_version=get_bundle()["version"])


@app.errorhandler(404)
def not_found(_):
    return jsonify(error="Endpoint not found"), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify(error="Method not allowed for this endpoint"), 405


@app.errorhandler(500)
def internal_error(_):
    return jsonify(error="Internal server error"), 500


# Load the model once when the module is imported (gunicorn does this per worker).
try:
    load_model()
except Exception:
    log.exception("Initial model load failed")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
