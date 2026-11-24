"""Unit tests for the Flask API endpoints."""

import json


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["model_version"].startswith("v")
    assert body["uptime_seconds"] >= 0


def test_predict_default_returns_7_days(client):
    resp = client.post("/predict")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["forecast_days"] == 7
    assert len(body["predictions"]) == 7
    first = body["predictions"][0]
    assert set(first) == {"date", "predicted_close"}
    assert first["predicted_close"] > 0


def test_predict_custom_days(client):
    resp = client.post("/predict", json={"forecast_days": 3})
    assert resp.status_code == 200
    assert len(resp.get_json()["predictions"]) == 3


def test_predict_with_history(client):
    history = [60000 + i * 100 for i in range(40)]
    resp = client.post("/predict", json={"history": history, "last_date": "2025-01-31"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["input_source"] == "request_history"
    assert body["predictions"][0]["date"] == "2025-02-01"


def test_predict_rejects_bad_json(client):
    resp = client.post("/predict", data="not json", content_type="application/json")
    assert resp.status_code == 400


def test_predict_rejects_non_object(client):
    resp = client.post("/predict", data=json.dumps([1, 2]), content_type="application/json")
    assert resp.status_code == 400


def test_predict_rejects_bad_forecast_days(client):
    for bad in (0, 8, "seven", 2.5, True):
        resp = client.post("/predict", json={"forecast_days": bad})
        assert resp.status_code == 400, bad


def test_predict_rejects_short_or_null_history(client):
    assert client.post("/predict", json={"history": [1, 2, 3]}).status_code == 400
    bad = [100.0] * 39 + [None]
    assert client.post("/predict", json={"history": bad}).status_code == 400
    neg = [100.0] * 39 + [-5]
    assert client.post("/predict", json={"history": neg}).status_code == 400


def test_predict_rejects_bad_date(client):
    history = [100.0 + i for i in range(40)]
    resp = client.post("/predict", json={"history": history, "last_date": "31-01-2025"})
    assert resp.status_code == 400


def test_metrics(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.get_json()
    for key in ("mae", "rmse", "mape"):
        assert key in body["metrics"]
    assert "api_stats" in body


def test_reload(client, api_module, monkeypatch):
    monkeypatch.setattr(api_module, "API_TOKEN", "secret123")
    assert client.post("/reload").status_code == 401
    resp = client.post("/reload", headers={"X-API-Token": "secret123"})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "reloaded"


def test_unknown_endpoint_returns_json_404(client):
    resp = client.get("/does-not-exist")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_predict_returns_500_on_model_failure(client, api_module, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(api_module, "forecast", boom)
    assert client.post("/predict").status_code == 500
