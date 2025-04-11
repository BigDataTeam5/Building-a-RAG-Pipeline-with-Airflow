FROM apache/airflow:2.10.4-python3.11

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

RUN mkdir -p /home/airflow/.cache && chmod 775 /home/airflow/.cache

COPY --chown=airflow:root Airflow/requirements.txt /opt/airflow/requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /opt/airflow/requirements.txt
