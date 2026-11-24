# Airflow image with our ML libraries added, so DAG tasks can fetch data
# and train the model.
# Only the ML libraries are installed, NOT app/requirements.txt: that file
# includes Flask 3 and Gunicorn for the API, and Airflow 2.10's own web UI
# needs Flask 2, so installing them here would break Airflow.
# numpy / pandas / scikit-learn / joblib versions match app/requirements.txt so
# a model trained by Airflow can be loaded by the API.
FROM apache/airflow:2.10.5-python3.10

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scikit-learn==1.5.2 \
    joblib==1.4.2 \
    "yfinance>=0.2.54" \
    "requests>=2.31" \
    pytest==8.3.4
