FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy source code
COPY src/ src/
COPY serving/ serving/

# Expose API port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run serving API
CMD ["python", "serving/api.py"]
