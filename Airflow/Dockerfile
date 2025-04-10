FROM apache/airflow:2.10.4

USER root

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget unzip gpg git build-essential \
    cmake python3-dev libpq-dev && \
    wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | \
    gpg --dearmor -o /usr/share/keyrings/google-chrome-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/google-chrome-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y --no-install-recommends google-chrome-stable && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /opt/airflow/chroma_db && \
    chown -R airflow:root /opt/airflow/chroma_db

# Install Rust compiler more efficiently
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# Switch to airflow user
USER airflow

# Copy just requirements first for better caching
COPY --chown=airflow:root Airflow/requirements.txt /tmp/requirements.txt

# Install dependencies with appropriate options
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Create .cache directory with correct permissions
RUN mkdir -p /home/airflow/.cache && chmod 775 /home/airflow/.cache