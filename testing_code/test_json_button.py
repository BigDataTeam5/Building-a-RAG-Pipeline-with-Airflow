import streamlit as st
import requests
import time
from dotenv import load_dotenv
load_dotenv()
# Airflow API URL
AIRFLOW_API_URL = "http://localhost:8080/api/v1"  # Replace with your Airflow API URL

# Placeholder for the JSON button
json_button_placeholder = st.empty()
rdbms_button = st.empty()

# Add the JSON button inside the placeholder
with json_button_placeholder.container():
    json_button = st.button("JSON Transformation", key="json_button")

with rdbms_button.container(): 
    rdbms_button = st.button("RDBMS Transformation", key="rdbms_button")

# Function to trigger the JSON transformation DAG
def trigger_additional_dag(dag_id):
    # Unpause the DAG
    unpause_response = requests.patch(
        f"{AIRFLOW_API_URL}/dags/{dag_id}",
        json={"is_paused": False},
        headers={"Content-Type": "application/json"},
        auth=("admin", "admin")
    )
    if unpause_response.status_code == 200:
        st.info(f"✅ DAG {dag_id} unpaused successfully.")
    else:
        st.error(f"❌ Failed to unpause DAG: {unpause_response.status_code} - {unpause_response.text}")
        return

    # Trigger the DAG
    response = requests.post(
        f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns",
        json={},
        headers={"Content-Type": "application/json"},
        auth=("admin", "admin")  # Explicitly use basic authentication
    )
    if response.status_code == 200:
        st.success(f"✅ {dag_id} pipeline triggered successfully!")
        dag_run_id = response.json()["dag_run_id"]

        # Check the status of the last DAG run
        while True:
            status_response = requests.get(
                f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns/{dag_run_id}",
                headers={"Content-Type": "application/json"},
                auth=("admin", "admin")
            )
            if status_response.status_code == 200:
                status = status_response.json()["state"]
                if status in ["success", "failed"]:
                    break
                time.sleep(10)  # Wait for 10 seconds before checking again
            else:
                st.error(f"❌ Failed to get DAG run status: {status_response.status_code} - {status_response.text}")
                return

        if status == "success":
            st.success(f"✅ {dag_id} pipeline completed successfully!")
        else:
            st.error(f"❌ {dag_id} pipeline failed.")
    else:
        st.error(f"❌ Failed to trigger pipeline: {response.status_code} - {response.text}")

# Handle JSON button click
if json_button:
    trigger_additional_dag("json_dbt_transformation")  # Replace with your JSON transformation DAG ID

if rdbms_button:
    trigger_additional_dag("rdmbs_dbt_transformation")  # Replace with your RDBMS transformation DAG ID