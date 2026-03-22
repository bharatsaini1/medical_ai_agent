# ============================================================
#  Dockerfile — Medical AI Agent
#  Multi-stage build: keeps final image small
# ============================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching — 
# only re-installs packages if requirements.txt changes)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create logs directory
RUN mkdir -p logs embeddings

# Expose Flask port
EXPOSE 5000

# Environment variables (overridden by platform env vars)
ENV FLASK_PORT=5000
ENV FLASK_DEBUG=False
ENV LOG_LEVEL=INFO

# Start the Flask server
CMD ["python", "api/app.py"]
