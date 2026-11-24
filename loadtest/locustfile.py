"""Locust load test: many fake users calling the API at the same time.

Headless run (prints stats, writes CSV files for your report):
    locust -f loadtest/locustfile.py --host http://localhost:30080 \
        --headless -u 100 -r 10 -t 3m --csv loadtest/results

Or with the web UI (charts you can screenshot): leave out --headless and open
http://localhost:8089
"""

import random

from locust import HttpUser, between, task

HISTORY = [60000 + random.uniform(-2000, 2000) for _ in range(60)]


class ForecastUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(5)
    def predict_default(self):
        self.client.post("/predict", json={"forecast_days": 7})

    @task(3)
    def predict_with_history(self):
        self.client.post("/predict", json={"history": HISTORY}, name="/predict (history)")

    @task(1)
    def health(self):
        self.client.get("/health")
