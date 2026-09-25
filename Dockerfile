# Lightweight Python Base
FROM python:3.11-slim

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU PyTorch first for fast layer caching
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir flask pillow numpy opencv-python gunicorn

# Copy project files
COPY . .

# Expose port (Render/Cloud uses PORT env var, defaults to 5000)
ENV PORT=5000
EXPOSE 5000

# Start server with gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app
