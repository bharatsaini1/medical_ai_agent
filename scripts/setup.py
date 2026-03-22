#!/usr/bin/env python3
# scripts/setup.py
# ============================================================
#  One-time setup script — run this BEFORE starting the server
#
#  What it does:
#  1. Validates all API keys from .env
#  2. Processes and cleans the medical dataset
#  3. Generates and saves document embeddings cache
#  4. Creates Pinecone index and uploads all vectors
#
#  You only need to run this once (or after updating the dataset).
#  After that, the server loads the cached data on startup.
#
#  Usage:
#    python scripts/setup.py
# ============================================================

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import config
from utils.logger import logger
from utils.data_processor import prepare_dataset, load_documents
from retriever.vector import VectorRetriever


def main():
    print("=" * 60)
    print("  Medical AI Agent — Setup Script")
    print("=" * 60)

    # Step 1: Validate environment
    print("\n[1/4] Validating API keys...")
    try:
        config.validate()
        print("  ✓ All API keys present")
    except EnvironmentError as e:
        print(f"  ✗ {e}")
        sys.exit(1)

    # Step 2: Process dataset
    print("\n[2/4] Processing medical dataset...")
    t = time.time()
    docs = prepare_dataset()
    print(f"  ✓ {len(docs)} documents prepared in {time.time()-t:.1f}s")
    print(f"  Sample: {docs[0]['name']} — {docs[0]['symptoms'][:60]}...")

    # Step 3: Generate embeddings + index in Pinecone
    print("\n[3/4] Indexing documents in Pinecone (first run may take ~2 min)...")
    t = time.time()
    vector_retriever = VectorRetriever()
    vector_retriever.index_documents(docs)
    print(f"  ✓ Pinecone indexing done in {time.time()-t:.1f}s")

    # Step 4: Verify
    print("\n[4/4] Verifying setup...")
    stats = vector_retriever.index.describe_index_stats()
    total_vectors = stats.get("total_vector_count", 0)
    print(f"  ✓ Pinecone index '{config.PINECONE_INDEX_NAME}' has {total_vectors} vectors")

    print("\n" + "=" * 60)
    print("  Setup complete! You can now start the server:")
    print("  python api/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
