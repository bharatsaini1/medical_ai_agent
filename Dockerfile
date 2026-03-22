# ============================================================
#  Dockerfile — Railway Production (Size Optimized)
#
#  Problem: Default PyTorch = 8.3GB (GPU version) — too large
#  Fix: Force CPU-only PyTorch = ~800MB
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── CRITICAL: Install CPU-only PyTorch FIRST ──
# Must happen before sentence-transformers
# otherwise pip auto-pulls full 8GB GPU torch
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Install all other packages ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project ──
COPY . .
RUN mkdir -p logs embeddings
RUN chmod +x startup.sh

ENV FLASK_PORT=5000
ENV FLASK_DEBUG=False
ENV LOG_LEVEL=INFO

EXPOSE 5000

CMD ["bash", "startup.sh"]
