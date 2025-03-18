import streamlit as st
import requests
import pandas as pd
import snowflake.connector
from datetime import datetime
import time
import json
import numpy as np
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import redis
load_dotenv()
# ✅ Streamlit UI
st.set_page_config(page_title="Quarter Finder & Snowflake Viewer", layout="wide")
st.sidebar.title("📊 Navigation")

# ✅ Correct FastAPI Endpoint
REDIS_HOST = os.getenv("REDIS_HOST")  # Docker service name
REDIS_PORT = os.getenv("REDIS_PORT")
print(f"🔍 DEBUG: Redis Host - {REDIS_HOST}, Port - {REDIS_PORT}")
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    redis_client.ping()  # Quick check to ensure Redis is reachable
    print("✅ Connected to Redis")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    redis_client = None  # Fallback if Redis isn't available
client = redis.Redis(host=REDIS_HOST, port=6379, db=0)  # match your config
client.delete("snowflake_table_list")              # or "table_data:<table_name>"
# or client.flushall()  # CAREFUL: This removes ALL keys in Redis!

API_URL = os.getenv("API_URL")
AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL")
# Check if API_URL is loaded correctly
if not API_URL:
    st.error("❌ API_URL environment variable is not set.")
    st.stop()

# ✅ Placeholder for buttons
json_button_placeholder = st.empty()
rdbms_button_placeholder = st.empty()
# Initialize session state for buttons
if 'json_button_enabled' not in st.session_state:
    st.session_state.json_button_enabled = False
if 'rdbms_button_enabled' not in st.session_state:
    st.session_state.rdbms_button_enabled = False

# ✅ Snowflake Configuration
SNOWFLAKE_CONN_ID = os.getenv("SNOWFLAKE_CONN_ID")  # This will now come from .env or docker-compose
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
SNOWFLAKE_LOGIN = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")

print(f"✅ Snowflake Config Loaded: {SNOWFLAKE_ACCOUNT}")
# ✅ Snowflake Connection Function

def get_snowflake_connection():
    """Establish a secure connection to Snowflake."""
    try:
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_LOGIN,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA,
            role=SNOWFLAKE_ROLE,
            client_session_keep_alive=True,
            login_timeout=60,
            autocommit=True
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to Snowflake: {e}")
        return None

# ✅ Fetch List of Schemas
def get_schema_list():
    try:
        conn = get_snowflake_connection()
        if conn:
            query = "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA"
            df = pd.read_sql(query, conn)
            conn.close()
            return df["SCHEMA_NAME"].tolist() if "SCHEMA_NAME" in df.columns else []
    except Exception as e:
        st.error(f"❌ Error fetching schema list: {e}")
        return []


# ✅ Fetch Table List
def get_table_list(schema):
    """Retrieve all tables in a specified schema, cached in Redis."""
    # 1) Build a cache key
    redis_key = f"snowflake_table_list:{schema}"

    # 2) Check if we have a cached list
    if redis_client:
        cached_list = redis_client.get(redis_key)
        if cached_list:
            print("✅ Retrieved table list from Redis cache")
            return json.loads(cached_list)

    # 3) If not cached, fetch from Snowflake
    conn = get_snowflake_connection()
    if conn:
        try:
            query = f"SELECT TABLE_NAME FROM {SNOWFLAKE_DATABASE}.INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = '{schema}'"
            df = pd.read_sql(query, conn)
            conn.close()

            if df.empty:
                st.warning("⚠️ No tables found in the selected schema.")
                return []

            # Convert "TABLE_NAME" column to a Python list
            table_list = df["TABLE_NAME"].tolist() if "TABLE_NAME" in df.columns else []

            # 4) Store in Redis (set a TTL if desired)
            if redis_client:
                redis_client.set(redis_key, json.dumps(table_list), ex=3600)  
                # ex=3600 sets 1-hour expiration

            return table_list
        except Exception as e:
            st.error(f"Error fetching table list: {e}")
            return []
    return []

def fetch_filtered_data(schema, table_name, offset=0, limit=5000, filters=None):
    """Retrieve filtered data from a table in a specified schema, with Redis cache."""
    # Convert filters to JSON serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        return obj

    if filters:
        filters = {k: convert_to_serializable(v) for k, v in filters.items()}

    # 1. Build a unique key for the cache
    redis_key = f"filtered_data:{schema}:{table_name}:{offset}:{limit}:{json.dumps(filters)}"

    # 2. Clear Redis cache (optional, use with caution)
    if redis_client:
        redis_client.flushall()
        print("✅ Redis cache cleared")

    # 3. Check if we already have cached data
    if redis_client:
        cached_data = redis_client.get(redis_key)
        if cached_data:
            try:
                # Decode from bytes to string
                cached_str = cached_data.decode("utf-8")
                # Convert JSON back to DataFrame
                df = pd.read_json(cached_str, orient="records")
                print(f"✅ Retrieved filtered data from Redis cache for table: {table_name}")
                return df
            except Exception as e:
                # If JSON parse fails, delete stale cache and fall back
                redis_client.delete(redis_key)
                st.warning(f"⚠️ Cache for '{redis_key}' was invalid, deleted. Re-fetching from Snowflake.")
                print(f"❌ Error reading cached data: {e}")

    # 4. If no cached data (or we just deleted it), fetch from Snowflake
    try:
        conn = get_snowflake_connection()
        if conn:
            # Base query
            query = f'SELECT * FROM {SNOWFLAKE_DATABASE}.{schema}."{table_name}"'

            # Apply filters in SQL query for efficiency
            where_clauses = []
            if filters:
                for column, value in filters.items():
                    if isinstance(value, list) and len(value) == 2:  # Date range filter
                        if column.lower() in ['ddate', 'filedate', 'created_dt']:  # Detect date fields
                            start_date = value[0].strftime('%Y-%m-%d')
                            end_date = value[1].strftime('%Y-%m-%d')
                            where_clauses.append(f'"{column}" BETWEEN \'{start_date}\' AND \'{end_date}\'')
                    elif isinstance(value, tuple):  # Numerical range filter
                        where_clauses.append(f'"{column}" BETWEEN {value[0]} AND {value[1]}')
                    elif value and value != "":  # Categorical selection
                        where_clauses.append(f'"{column}" = \'{value}\'')
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            query += f' LIMIT {limit} OFFSET {offset}'

            df = pd.read_sql(query, conn)
            conn.close()

            # Convert date columns to datetime
            date_columns = ['ddate', 'filedate', 'created_dt']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            # Log the data types of the columns
            print(df.dtypes)

            # 5. Store in Redis for next time (set a TTL if desired)
            if redis_client:
                redis_client.set(redis_key, df.to_json(orient="records"), ex=3600)
                print(f"✅ Cached filtered data in Redis for table: {table_name}")

            return df if not df.empty else pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error fetching filtered data: {e}")
        return pd.DataFrame()
    
# ✅ Fetch Table Data
def fetch_table_data(table_name, filters=None):
    """Retrieve up to 100 rows from a table in SEC_SCHEMA, with Redis cache."""
    # 1. Build a unique key for the cache
    redis_key = f"table_data:{table_name}"

    # 2. Check if we already have cached data
    if redis_client:
        cached_data = redis_client.get(redis_key)
        if cached_data:
            try:
                # Decode from bytes to string
                cached_str = cached_data.decode("utf-8")
                # Convert JSON back to DataFrame
                df = pd.read_json(cached_str, orient="records")
                print(f"✅ Retrieved data from Redis cache for table: {table_name}")
                return df
            except Exception as e:
                # If JSON parse fails, delete stale cache and fall back
                redis_client.delete(redis_key)
                st.warning(f"⚠️ Cache for '{redis_key}' was invalid, deleted. Re-fetching from Snowflake.")
                print(f"❌ Error reading cached data: {e}")


    # 3. If no cached data (or we just deleted it), fetch from Snowflake
    conn = get_snowflake_connection()
    if conn:
        try:
            query = f'SELECT * FROM {SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}."{table_name}" LIMIT 100'
            df = pd.read_sql(query, conn)
            conn.close()

            # Apply filters if provided
            if filters:
                for column, filter_value in filters.items():
                    if isinstance(filter_value, tuple) and len(filter_value) == 2:
                        df = df[(df[column] >= filter_value[0]) & (df[column] <= filter_value[1])]
                    elif filter_value:
                        df = df[df[column] == filter_value]

            # 4. Store in Redis for next time (set a TTL if desired)
            if redis_client:
                redis_client.set(redis_key, df.to_json(orient="records"), ex=3600)
                print(f"✅ Cached data in Redis for table: {table_name}")

            return df
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# ✅ Execute Custom SQL Query
def execute_query(query):
    """Execute a custom SELECT query on Snowflake, cached in Redis."""
    # 1) Normalize or sanitize query if needed (lowercase, strip spaces, etc.)
    normalized_query = query.strip().lower()

    if not normalized_query.startswith("select"):
        st.error("❌ Only SELECT queries are allowed for security reasons.")
        return pd.DataFrame()

    # 2) Build the cache key (you could hash the query for cleanliness)
    redis_key = f"custom_query:{normalized_query}"

    # 3) Check Redis
    if redis_client:
        cached_data = redis_client.get(redis_key)
        if cached_data:
            print("✅ Retrieved query result from Redis cache")
            return pd.read_json(cached_data, orient="records")

    # 4) If no cache, run the query
    conn = get_snowflake_connection()
    if not conn:
        return pd.DataFrame()
    try:
        df = pd.read_sql(query, conn)
        conn.close()

        # 5) Store result in Redis
        if redis_client and not df.empty:
            redis_client.set(redis_key, df.to_json(orient="records"), ex=3600)
            print("✅ Cached custom query result in Redis")

        return df
    except Exception as e:
        st.error(f"❌ Query Execution Failed: {e}")
        return pd.DataFrame()


# ✅ Function to trigger Airflow DAG with year_quarter and check status
def trigger_airflow_dag(year_quarter):
    dag_id = "selenium_sec_pipeline"  # Replace with your DAG ID

    # Construct the URL for unpausing the DAG
    unpause_url = f"{AIRFLOW_API_URL}/dags/{dag_id}"
    # Construct the URL for triggering the DAG
    trigger_url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns"

    # Unpause the DAG
    unpause_response = requests.patch(
        unpause_url,
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
    payload = {
        "conf": {"year_quarter": year_quarter}  # ✅ Pass quarter directly
    }
    
    response = requests.post(
        trigger_url,
        json=payload,
        headers={"Content-Type": "application/json"},
        auth=("admin", "admin")  # ✅ Explicitly use basic authentication
    )
    if response.status_code == 200:
        st.success(f"✅ Pipeline triggered for Quarter: {year_quarter}")
        dag_run_id = response.json()["dag_run_id"]
        
        # Check the status of the last DAG run
        while True:
            status_response = requests.get(
                f"{trigger_url}/{dag_run_id}",
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
            st.success(f"✅ Pipeline for Quarter {year_quarter} completed successfully!")
            # ✅ Enable the buttons
            st.session_state.json_button_enabled = True
            st.session_state.rdbms_button_enabled = True
            st.rerun()
        else:
            st.error(f"❌ Pipeline for Quarter {year_quarter} failed.")    
    else:
        st.error(f"❌ Failed to trigger pipeline: {response.status_code} - {response.text}")

# ✅ Function to trigger additional Airflow DAGs
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
        auth=("admin", "admin")  # ✅ Explicitly use basic authentication
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

# Add buttons inside placeholders (initially disabled)
with json_button_placeholder.container():
    json_button = st.button("JSON Transformation", disabled=not st.session_state.json_button_enabled, key="json_button")
with rdbms_button_placeholder.container():
    rdbms_button = st.button("RDBMS Transformation", disabled=not st.session_state.rdbms_button_enabled, key="rdbms_button")
# ✅ Handle JSON button click
if json_button:
    trigger_additional_dag("json_dbt_transformation")  # Replace with your JSON transformation DAG ID

# ✅ Handle RDBMS button click
if rdbms_button:
    trigger_additional_dag("rdmbs_dbt_transformation")  # Replace with your RDBMS transformation DAG ID
    
# ✅ Sidebar - Select Schema
schemas = get_schema_list()
if schemas:
    SNOWFLAKE_SCHEMA = st.sidebar.selectbox("Select Schema", schemas)
else:
    st.error("❌ No schemas found. Check database connection or permissions.")
    SNOWFLAKE_SCHEMA = None

# ✅ Sidebar - Select View
view_option = st.sidebar.radio("Choose View:", ["Find Quarter", "View Snowflake Tables", "Query Snowflake Table", "Visualizations"])

# ✅ Quarter Finder Feature
if view_option == "Find Quarter":
    st.title("📆 Quarter Finder API")
    st.markdown("Enter a date (YYYY-MM-DD) to find the corresponding **year and quarter**.")
    
    date_input = st.date_input("Enter Date (YYYY-MM-DD)", datetime(2023, 6, 15), min_value=datetime(2000, 1, 1))
    
    if st.button("Get Quarter"):
        date_str = date_input.strftime("%Y-%m-%d")
        st.info(f"📅 Finding quarter for date: **{date_str}**")
        # Send POST request to FastAPI backend
        response = requests.post(API_URL, json={"date": date_str})

        if response.status_code == 200:
            result = response.json()
            year_quarter = result["year_quarter"]
            st.success(f"🗓 The corresponding year and quarter: **{result['year_quarter']}**")
            st.info(f"Updated config with date: {date_str}")  # Add logging to verify the update
            # ✅ Directly send the quarter to Airflow DAG
            trigger_airflow_dag(year_quarter)
        else:
            st.error(f"⚠️ API Error: {response.status_code} - {response.text}")

# ✅ Snowflake Table Viewer Feature with Column Filters
elif view_option == "View Snowflake Tables" and SNOWFLAKE_SCHEMA:
    st.title("📂 Snowflake Table Viewer")
    tables = get_table_list(SNOWFLAKE_SCHEMA)

    if tables:
        selected_table = st.sidebar.selectbox("Select a Table", tables)

        if selected_table:
            # Fetch FULL data to get filter options
            full_df = fetch_filtered_data(SNOWFLAKE_SCHEMA, selected_table, limit=5000)

            if not full_df.empty:
                st.sidebar.subheader("🎯 Column Filters")

                # Ensure columns are stripped of whitespace and converted to lowercase for consistent filtering
                excluded_columns = {'cik', 'ein', 'changed','value'}  # Add other unwanted columns here
                filtered_columns = [
                    col for col in full_df.columns
                    if not (col.lower().strip() in excluded_columns or 
                            col.lower().strip().endswith(('_sk', '_dt', '_id', '_code')))
                ]

                filters = {}
                for column in filtered_columns:
                    unique_values = full_df[column].dropna().unique()
                    if len(unique_values) < 15:  # Categorical filter
                        filters[column] = st.sidebar.selectbox(f"Filter {column}", [""] + list(unique_values), key=column)
                    elif pd.api.types.is_numeric_dtype(full_df[column]):  # Numerical range filter
                        min_val, max_val = int(full_df[column].min()), int(full_df[column].max())
                        filters[column] = st.sidebar.slider(f"Filter {column}", min_val, max_val, (min_val, max_val), key=column)
                    elif pd.api.types.is_datetime64_any_dtype(full_df[column]):  # Date range filter
                        min_date, max_date = full_df[column].min(), full_df[column].max()
                        filters[column] = st.sidebar.date_input(f"Filter {column}", [min_date, max_date], key=column)

                # Apply button for filters
                apply_filters = st.sidebar.button("Apply Filters")

                # Fetch filtered data only if "Apply Filters" is clicked
                if apply_filters:
                    filtered_df = fetch_filtered_data(SNOWFLAKE_SCHEMA, selected_table, filters=filters)

                    if not filtered_df.empty:
                        st.subheader(f"📄 Filtered Data from `{selected_table}`")
                        st.data_editor(filtered_df)  # Efficient DataFrame rendering
                    else:
                        st.warning("⚠️ No data available with the applied filters.")
            else:
                st.warning("⚠️ No data available in the selected table.")
    else:
        st.error("⚠️ No tables found. Check your **database connection** or **permissions**.")

elif view_option == "Query Snowflake Table":
    st.title("📝 Execute Custom SQL Query on Snowflake")
    query = st.text_area("Enter your SQL query (Only SELECT queries allowed)", "SELECT * FROM PUBLIC.SAMPLE_TABLE LIMIT 10")

    if st.button("Run Query"):
        df = execute_query(query)

        if not df.empty:
            st.dataframe(df)
        else:
            st.warning("⚠️ No data returned from query.")


# ✅ Data Visualization
elif view_option == "Visualizations" and SNOWFLAKE_SCHEMA:
    st.title("📊 Data Visualization")
    tables = get_table_list(SNOWFLAKE_SCHEMA)
    if tables:
        selected_table = st.sidebar.selectbox("Select a Table for Visualization", tables)
        if selected_table:
            sample_df = fetch_filtered_data(SNOWFLAKE_SCHEMA, selected_table, limit=1000)
            if not sample_df.empty:
                numeric_columns = sample_df.select_dtypes(include=["number"]).columns.tolist()
                categorical_columns = sample_df.select_dtypes(include=["object"]).columns.tolist()
                
                if numeric_columns or categorical_columns:
                    plot_type = st.sidebar.radio("Select Plot Type", ["Scatter Plot", "Line Chart", "Bar Chart", "Pie Chart", "Histogram"])

                    fig, ax = plt.subplots(figsize=(8, 5))
                    if plot_type == "Scatter Plot" and numeric_columns:
                        x_column = st.sidebar.selectbox("Select X-axis Column", numeric_columns)
                        y_column = st.sidebar.selectbox("Select Y-axis Column", numeric_columns)
                        ax.scatter(sample_df[x_column], sample_df[y_column])
                    elif plot_type == "Line Chart" and numeric_columns:
                        x_column = st.sidebar.selectbox("Select X-axis Column", numeric_columns)
                        y_column = st.sidebar.selectbox("Select Y-axis Column", numeric_columns)
                        ax.plot(sample_df[x_column], sample_df[y_column])
                    elif plot_type == "Bar Chart" and categorical_columns:
                        cat_column = st.sidebar.selectbox("Select Categorical Column", categorical_columns)
                        num_column = st.sidebar.selectbox("Select Numerical Column", numeric_columns)
                        ax.bar(sample_df[cat_column].astype(str), sample_df[num_column])
                    elif plot_type == "Pie Chart" and categorical_columns:
                        cat_column = st.sidebar.selectbox("Select Categorical Column for Pie Chart", categorical_columns)
                        sample_df[cat_column].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
                    elif plot_type == "Histogram" and numeric_columns:
                        hist_column = st.sidebar.selectbox("Select Column for Histogram", numeric_columns)
                        sample_df[hist_column].plot(kind='hist', ax=ax)

                    st.pyplot(fig)
                else:
                    st.warning("⚠️ No numeric or categorical columns available for visualization.")
            else:
                st.warning("⚠️ No data available in the selected table.")
    else:
        st.error("⚠️ No tables found. Check your **database connection** or **permissions**.")