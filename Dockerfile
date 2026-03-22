FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p logs embeddings
RUN chmod +x startup.sh

ENV FLASK_PORT=5000
ENV FLASK_DEBUG=False
ENV LOG_LEVEL=INFO

EXPOSE 5000

# Just start Flask directly — no setup.py at runtime
CMD ["python", "api/app.py"]
