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
RUN pip install -r requirements.txt

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY main.py .

# Note: picsellia_token and Readme.md are optional
# They can be mounted as volumes if not in build context

# Default command
CMD ["python", "main.py", "train", "--evaluate"]
