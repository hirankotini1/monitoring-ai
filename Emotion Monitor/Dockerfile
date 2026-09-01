# Multi-Platform Linux Container for Render, Docker & Cloud
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

# Install native system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose server port
EXPOSE 5000

# Start server entrypoint
CMD ["python", "server.py"]
