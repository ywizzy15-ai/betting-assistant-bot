FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port (Render injects $PORT env var)
EXPOSE 8080

# Run gunicorn with exact command (overrides any Procfile)
CMD ["gunicorn", "main:app", \
     "--bind", "0.0.0.0:8080", \
     "--worker-class", "aiohttp.worker.GunicornWebWorker", \
     "--workers", "2", \
     "--timeout", "30"]

