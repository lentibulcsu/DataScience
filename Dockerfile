# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports (8000 for FastAPI, 8888 for Jupyter)
EXPOSE 8000 8888

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command runs the FastAPI web app
# To run Jupyter instead, override with: docker run -p 8888:8888 <image> jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
