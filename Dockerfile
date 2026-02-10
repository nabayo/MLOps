# syntax=docker/dockerfile:1
FROM python:3.12-slim-trixie

WORKDIR /app

# Install system dependencies (OpenCV, etc.)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with pip cache mount for faster rebuilds
# --mount=type=cache uses BuildKit cache to persist pip cache between builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --ignore-installed -r requirements.txt

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY main.py .

# Note: picsellia_token and Readme.md are optional
# They can be mounted as volumes if not in build context

# Default command
CMD ["python", "main.py", "train", "--evaluate"]
