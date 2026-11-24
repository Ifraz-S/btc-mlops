# Airflow image with our ML libraries added, so DAG tasks can fetch data
# and train the model. Same library versions as the API image.
FROM apache/airflow:2.10.5-python3.10

COPY app/requirements.txt /tmp/app-requirements.txt
RUN pip install --no-cache-dir -r /tmp/app-requirements.txt "yfinance>=0.2.54" "requests>=2.31" pytest==8.3.4
