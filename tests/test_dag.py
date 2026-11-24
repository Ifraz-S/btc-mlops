"""DAG integrity test. Only runs where Airflow is installed (e.g. inside the
Airflow container: docker compose exec airflow-scheduler pytest /opt/airflow/tests).
It is skipped in normal CI, which does not install Airflow."""

from pathlib import Path

import pytest

pytest.importorskip("airflow.models")
from airflow.models import DagBag  # noqa: E402

DAGS_DIR = Path(__file__).resolve().parent.parent / "dags"


def test_dag_loads_without_errors():
    bag = DagBag(dag_folder=str(DAGS_DIR), include_examples=False)
    assert bag.import_errors == {}
    dag = bag.dags.get("btc_etl_pipeline")
    assert dag is not None
    assert {t.task_id for t in dag.tasks} == {
        "fetch_btc_data",
        "preprocess_features",
        "train_and_evaluate",
        "check_model_quality",
        "register_model",
        "skip_registration",
        "reload_api_model",
    }
