# Use official Apache Airflow image
FROM apache/airflow:2.6.3

# Set Airflow Home directory
ENV AIRFLOW_HOME=/opt/airflow
WORKDIR $AIRFLOW_HOME

# Switch to root user temporarily
USER root

# Copy the requirements file
COPY requirements.txt $AIRFLOW_HOME/requirements.txt

# Install dependencies as airflow user
RUN chown -R airflow: $AIRFLOW_HOME

# Switch back to airflow user
USER airflow

# Install dependencies
RUN pip install --no-cache-dir -r $AIRFLOW_HOME/requirements.txt

# Ensure correct paths when copying DAGs
COPY . /opt/airflow/dags/

# Expose Airflow webserver port
EXPOSE 8080

# Default command to run Airflow scheduler
CMD ["airflow", "scheduler"]
