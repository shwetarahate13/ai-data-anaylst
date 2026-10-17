FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and writing pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Generate demo database on build
RUN python -m analyst generate-data

# Expose FastAPI (8000)
EXPOSE 8000

# Default entrypoint runs production FastAPI service (serving both UI and API)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
