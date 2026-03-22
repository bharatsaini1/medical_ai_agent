# retriever/vector.py
# ============================================================
#  Vector Retriever — Pinecone v6 + sentence-transformers
#
#  Pinecone v6 key changes vs old pinecone-client v4:
#  - Package is now just `pinecone` (not `pinecone-client`)
#  - describe_index_stats() returns an object, not a dict
#  - response.matches is a list of Match objects (not dicts)
#  - Everything else (upsert format, query format) is the same
# ============================================================

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from utils.config import config
from utils.logger import logger


class VectorRetriever:
    """
    Handles embedding generation, Pinecone index creation/population,
    and similarity search.
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.encoder = SentenceTransformer(config.EMBEDDING_MODEL)

        logger.info("Connecting to Pinecone...")
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = self._get_or_create_index()

    def _get_or_create_index(self):
        """Get existing Pinecone index or create a new one."""
        index_name = config.PINECONE_INDEX_NAME

        # pinecone v6: list_indexes() returns IndexList object
        existing = self.pc.list_indexes()

        # Support both v6 object style and v4 dict style
        try:
            if hasattr(existing, 'indexes'):
                existing_names = [idx.name for idx in existing.indexes]
            else:
                existing_names = [
                    idx.name if hasattr(idx, 'name') else idx.get('name', '')
                    for idx in existing
                ]
        except Exception:
            existing_names = []

        if index_name not in existing_names:
            logger.info(f"Creating Pinecone index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=config.PINECONE_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info(f"Pinecone index '{index_name}' created")
        else:
            logger.info(f"Pinecone index '{index_name}' already exists")

        return self.pc.Index(index_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate normalized embeddings."""
        return self.encoder.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def _get_vector_count(self) -> int:
        """Safely get current vector count — handles v4 dict and v6 object."""
        try:
            stats = self.index.describe_index_stats()
            if hasattr(stats, 'total_vector_count'):
                return stats.total_vector_count or 0
            if isinstance(stats, dict):
                return stats.get('total_vector_count', 0)
        except Exception as e:
            logger.warning(f"Could not get vector count: {e}")
        return 0

    def index_documents(self, documents: list[dict], batch_size: int = 100):
        """
        Embed all documents and upsert into Pinecone.
        Skips if already indexed.
        """
        existing_count = self._get_vector_count()

        if existing_count >= len(documents):
            logger.info(
                f"Pinecone already has {existing_count} vectors "
                f"({len(documents)} in dataset). Skipping re-index."
            )
            return

        logger.info(f"Indexing {len(documents)} documents into Pinecone...")

        texts = [doc["text"] for doc in documents]
        embeddings = self.encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        for i in tqdm(range(0, len(documents), batch_size), desc="Upserting to Pinecone"):
            batch_docs = documents[i: i + batch_size]
            batch_embs = embeddings[i: i + batch_size]

            vectors = [
                {
                    "id":     doc["id"],
                    "values": emb.tolist(),
                    "metadata": {
                        "name":         doc["name"],
                        "symptoms":     doc["symptoms"],
                        "treatments":   doc["treatments"],
                        "contagious":   doc["contagious"],
                        "chronic":      doc["chronic"],
                        "disease_code": doc["disease_code"],
                        "text":         doc["text"],
                    },
                }
                for doc, emb in zip(batch_docs, batch_embs)
            ]

            self.index.upsert(vectors=vectors)

        logger.info("Pinecone indexing complete")

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """Retrieve top-k semantically similar documents."""
        if top_k is None:
            top_k = config.TOP_K

        query_vec = self.embed([query])[0].tolist()

        response = self.index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True,
        )

        # v6: response.matches is a list of Match objects
        matches = getattr(response, 'matches', None) or response.get('matches', [])

        results = []
        for match in matches:
            # Support both object (v6) and dict (v4)
            if hasattr(match, 'metadata'):
                meta  = match.metadata or {}
                score = float(match.score)
                mid   = match.id
            else:
                meta  = match.get('metadata', {})
                score = float(match.get('score', 0.0))
                mid   = match.get('id', '')

            results.append({
                "doc": {
                    "id":           mid,
                    "name":         meta.get("name", ""),
                    "symptoms":     meta.get("symptoms", ""),
                    "treatments":   meta.get("treatments", ""),
                    "contagious":   meta.get("contagious", ""),
                    "chronic":      meta.get("chronic", ""),
                    "disease_code": meta.get("disease_code", ""),
                    "text":         meta.get("text", ""),
                },
                "score":            score,
                "normalised_score": score,
            })

        logger.debug(f"Vector search: {len(results)} results for '{query}'")
        return results
