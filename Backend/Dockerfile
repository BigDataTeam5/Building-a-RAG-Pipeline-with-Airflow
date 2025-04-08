FROM python:3.10-slim

# Prevent Python from creating .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    dos2unix \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

RUN poetry config virtualenvs.create false

# Set working directory
WORKDIR /app

# Copy project configuration files from root
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-root --no-interaction --no-ansi

# Create required directories
RUN mkdir -p /app/uploads \
    /app/user_markdowns \
    /app/chunk_storage \
    /app/local_vector.db

# Copy application code
COPY Backend/ /app/Backend/
COPY Airflow/ /app/Airflow/
COPY Rag_modelings/ /app/Rag_modelings/
COPY .env /app/.env

# Add entrypoint script
COPY Backend/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
RUN dos2unix /app/entrypoint.sh

# Convert line endings
RUN find /app -name "*.py" -exec dos2unix {} \;

# Use entrypoint script
CMD ["/app/entrypoint.sh"]