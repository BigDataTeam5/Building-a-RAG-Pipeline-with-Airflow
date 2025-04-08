# Use the official Apache Airflow image
FROM apache/airflow:2.10.4

# Switch to root user to install system dependencies
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    wget \
    gpg \
    git \
    build-essential \
    cmake \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Add Google's signing key & repo, then install Chrome
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/google-chrome-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install Rust compiler (needed for tokenizers package)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create directories for ChromaDB and set permissions
RUN mkdir -p /opt/airflow/chroma_db && \
    chown -R airflow:root /opt/airflow/chroma_db

# Set cargo environment for the airflow user
RUN mkdir -p /home/airflow/.cargo && \
    chown -R airflow:root /home/airflow/.cargo
ENV CARGO_HOME=/home/airflow/.cargo
ENV RUSTUP_HOME=/home/airflow/.rustup

# Switch to airflow user
USER airflow

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# First install huggingface-hub at exactly the right version
RUN pip install --no-cache-dir huggingface-hub==0.16.4

# Then install transformers and related packages
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    safetensors \
    accelerate \
    scikit-learn

# Then install sentence-transformers with --no-deps to prevent it from updating huggingface-hub
RUN pip install --no-cache-dir --no-deps sentence-transformers==2.2.2
# Finally install other packages from requirements
COPY --chown=airflow:root Airflow/requirements.txt /tmp/requirements.txt
# Skip reinstalling packages we've already installed
RUN grep -v "huggingface-hub\|sentence-transformers\|transformers\|safetensors\|accelerate" /tmp/requirements.txt > /tmp/filtered_requirements.txt
RUN pip install --no-cache-dir -r /tmp/filtered_requirements.txt

# Stay as airflow user
USER airflow