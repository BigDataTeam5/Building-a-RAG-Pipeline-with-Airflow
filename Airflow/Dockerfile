FROM apache/airflow:2.10.4

USER root

# Install system dependencies and Chrome
RUN apt-get update && apt-get install -y \
    curl wget unzip gpg git build-essential \
    cmake python3-dev libpq-dev && \
    wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | \
    gpg --dearmor -o /usr/share/keyrings/google-chrome-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/google-chrome-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /opt/airflow/chroma_db && \
    chown -R airflow:root /opt/airflow/chroma_db

# Install Rust compiler 
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

USER airflow

# Install Python dependencies with specific versions to ensure compatibility
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir huggingface-hub==0.16.4 && \
    pip install --no-cache-dir transformers==4.36.2 safetensors && \
    pip install --no-cache-dir --no-deps sentence-transformers==2.2.2 && \
    pip install --no-cache-dir accelerate==0.27.0 scikit-learn

# Copy and install other requirements
COPY --chown=airflow:root Airflow/requirements.txt /tmp/requirements.txt
RUN grep -v "huggingface-hub\|sentence-transformers\|transformers\|safetensors\|accelerate" /tmp/requirements.txt > /tmp/filtered_requirements.txt && \
    pip install --no-cache-dir -r /tmp/filtered_requirements.txt