FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements FIRST (before source code) for better caching
COPY requirements.txt .

# Install Python dependencies with pip cache mount for faster rebuilds
# --mount=type=cache uses BuildKit cache to persist pip cache between builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY serving/ serving/

# Expose API port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run serving API
CMD ["python", "serving/api.py"]
