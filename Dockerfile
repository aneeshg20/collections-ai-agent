# Start from official Python 3.12 slim image (smaller than full)
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer if requirements don't change)
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the rest of the project
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Health check so Docker knows when the app is ready
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app
ENTRYPOINT ["streamlit", "run", "agent/app_v2.py", "--server.port=8501", "--server.address=0.0.0.0"]
