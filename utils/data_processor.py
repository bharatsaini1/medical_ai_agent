# utils/data_processor.py
# ============================================================
#  Dataset cleaning and document preparation
#
#  Steps:
#  1. Load CSV (your real Diseases_Symptoms.csv)
#  2. Fill / drop missing values
#  3. Combine all fields into a single "document" string
#     — this is what gets embedded and indexed
#  4. Return a list of dicts ready for BM25 + Pinecone
# ============================================================

import os
import json
import pandas as pd
from utils.config import config
from utils.logger import logger


def load_and_clean() -> pd.DataFrame:
    """
    Load the CSV and clean it:
    - Strip whitespace from string columns
    - Fill missing Treatments with a placeholder
    - Drop rows that are entirely empty
    """
    logger.info(f"Loading dataset from: {config.DATA_PATH}")
    df = pd.read_csv(config.DATA_PATH)

    logger.info(f"Raw shape: {df.shape}")
    logger.info(f"Null counts before cleaning:\n{df.isnull().sum()}")

    # Strip leading/trailing whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Fill the 1 missing Treatment value
    df["Treatments"] = df["Treatments"].fillna("Treatment information not available")

    # Ensure boolean columns are proper booleans
    df["Contagious"] = df["Contagious"].astype(bool)
    df["Chronic"] = df["Chronic"].astype(bool)

    # Drop any fully empty rows just in case
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Clean shape: {df.shape}")
    return df


def build_documents(df: pd.DataFrame) -> list[dict]:
    """
    Convert each row into a rich document dict.

    The 'text' field combines all relevant fields into ONE string.
    This is what BM25 searches over and what gets embedded.

    Format:
        "Disease: <name> | Symptoms: <symptoms> | Treatment: <treatment>
         | Contagious: Yes/No | Chronic: Yes/No | Code: <code>"

    Why one string?
    - BM25 searches plain text — needs everything in one place
    - Embeddings capture semantic meaning of the full context
    - Easier to display to users as a unified chunk
    """
    documents = []
    for _, row in df.iterrows():
        contagious_str = "Yes" if row["Contagious"] else "No"
        chronic_str    = "Yes" if row["Chronic"]    else "No"

        combined_text = (
            f"Disease: {row['Name']} | "
            f"Symptoms: {row['Symptoms']} | "
            f"Treatment: {row['Treatments']} | "
            f"Contagious: {contagious_str} | "
            f"Chronic: {chronic_str} | "
            f"Code: {row['Disease_Code']}"
        )

        documents.append({
            "id":           row["Disease_Code"],        # unique ID for Pinecone
            "name":         row["Name"],
            "symptoms":     row["Symptoms"],
            "treatments":   row["Treatments"],
            "contagious":   contagious_str,
            "chronic":      chronic_str,
            "disease_code": row["Disease_Code"],
            "text":         combined_text,              # used for BM25 + embedding
        })

    logger.info(f"Built {len(documents)} documents")
    return documents


def save_documents(documents: list[dict]):
    """
    Save processed documents to disk as JSON.
    This cache avoids re-processing on every startup.
    """
    os.makedirs(os.path.dirname(config.DOCUMENTS_PATH), exist_ok=True)
    with open(config.DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)
    logger.info(f"Saved {len(documents)} documents to {config.DOCUMENTS_PATH}")


def load_documents() -> list[dict]:
    """
    Load cached documents from disk.
    Falls back to re-building from CSV if cache doesn't exist.
    """
    if os.path.exists(config.DOCUMENTS_PATH):
        with open(config.DOCUMENTS_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        logger.info(f"Loaded {len(docs)} cached documents from {config.DOCUMENTS_PATH}")
        return docs

    logger.warning("Document cache not found. Re-building from CSV...")
    df = load_and_clean()
    docs = build_documents(df)
    save_documents(docs)
    return docs


def prepare_dataset():
    """
    Main entry point: clean data, build documents, save to disk.
    Call this once during project setup (or from scripts/setup.py).
    """
    df = load_and_clean()
    docs = build_documents(df)
    save_documents(docs)
    return docs


if __name__ == "__main__":
    # Quick sanity check: run this file directly to see the output
    docs = prepare_dataset()
    print(f"\nSample document:\n{json.dumps(docs[0], indent=2)}")
