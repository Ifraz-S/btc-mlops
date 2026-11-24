"""Airflow DAG: daily BTC ETL -> features -> retrain -> register -> reload API.

Task flow (what you see in the Airflow "Graph" view):

    fetch_btc_data
          |
    preprocess_features
          |
    train_and_evaluate
          |
    check_model_quality  (branch: did MAPE beat the threshold?)
        /          \\
  register_model   skip_registration
        |
  reload_api_model

The heavy lifting lives in src/ so the same code also runs outside Airflow.
"""

import json
import os
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

from src import config
from src.etl_fetch import fetch_btc_data
from src.features import run_preprocessing
from src.train_forecast import register_model, train_and_evaluate

API_URL = os.getenv("BTC_API_URL", "http://api:5000")
API_TOKEN = os.getenv("API_TOKEN", "")


def _fetch(**_):
    path = fetch_btc_data()
    return str(path)


def _preprocess(**_):
    return str(run_preprocessing())


def _train(**_):
    report = train_and_evaluate()
    # The return value is stored in XCom so the next task can read it.
    return {
        "model_version": report["model_version"],
        "mape": report["metrics"]["mape"],
        "passed": report["passed_threshold"],
    }


def _choose_branch(ti, **_):
    result = ti.xcom_pull(task_ids="train_and_evaluate")
    print(f"Candidate model result: {json.dumps(result)} (threshold {config.MAPE_THRESHOLD}%)")
    return "register_model" if result["passed"] else "skip_registration"


def _register(**_):
    if not register_model():
        raise RuntimeError("Registration failed")


def _reload_api(**_):
    resp = requests.post(f"{API_URL}/reload", headers={"X-API-Token": API_TOKEN}, timeout=30)
    print(f"API responded {resp.status_code}: {resp.text}")
    resp.raise_for_status()


default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="btc_etl_pipeline",
    description="Daily BTC-USD ETL, feature engineering, retraining and API reload",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "bitcoin", "forecast"],
) as dag:
    fetch = PythonOperator(task_id="fetch_btc_data", python_callable=_fetch)
    preprocess = PythonOperator(task_id="preprocess_features", python_callable=_preprocess)
    train = PythonOperator(task_id="train_and_evaluate", python_callable=_train)
    check = BranchPythonOperator(task_id="check_model_quality", python_callable=_choose_branch)
    register = PythonOperator(task_id="register_model", python_callable=_register)
    skip = EmptyOperator(task_id="skip_registration")
    reload_api = PythonOperator(task_id="reload_api_model", python_callable=_reload_api)

    fetch >> preprocess >> train >> check >> [register, skip]
    register >> reload_api
