#!/bin/bash
# startup.sh — Cloud deployment startup script
# 
# This runs ONCE on every container start:
# 1. Runs setup.py to index Pinecone (skips if already indexed)
# 2. Starts the Flask server
#
# Used by Railway / Render / Docker CMD

set -e  # exit immediately on any error

echo "=========================================="
echo "  Medical AI Agent — Starting Up"
echo "=========================================="

# Step 1: Run setup (safe to re-run — skips if Pinecone already indexed)
echo "[1/2] Running setup..."
python scripts/setup.py

# Step 2: Start Flask server
echo "[2/2] Starting Flask server..."
python api/app.py
