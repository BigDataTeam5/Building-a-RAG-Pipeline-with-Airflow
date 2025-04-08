FROM python:3.10-slim

# Prevent Python from creating .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system packages: dos2unix (for line ending fixes), curl (to install Poetry)
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

# Set our working directory
WORKDIR /app

# Copy project configuration files from root
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-root --no-interaction --no-ansi

# Copy all necessary directories and files except Frontend
COPY Backend/ /app/Backend/
COPY Airflow/ /app/Airflow/
COPY uploads/ /app/uploads/
COPY user_markdowns/ /app/user_markdowns/
COPY Rag_modelings/ /app/Rag_modelings/
COPY chunk_storage/ /app/chunk_storage/
COPY local_vector.db/ /app/local_vector.db

COPY .env /app/.env

# Convert line endings
RUN find /app -name "*.py" -exec dos2unix {} \;

# Change the CMD to use the correct path
CMD ["uvicorn", "Backend.main:app", "--host", "0.0.0.0", "--port", "8000"]