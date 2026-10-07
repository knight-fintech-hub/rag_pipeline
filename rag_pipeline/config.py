"""
Shared Configuration — RAG Pipeline
-------------------------------------
Single source of truth for all default constants used across:
  - ingestion.py  (DocumentIngestionPipeline)
  - embedding.py  (Embedding)
  - vector_store.py (QdrantVectorStore)

Import from here instead of redeclaring constants in each module.
"""

import os
import torch

# ── Embedding model ──────────────────────────────────────────────────
DEFAULT_EMBEDDING_MODEL: str = r"E:\ds-team\huggingface_models\Old BKP\test_models\bge_m3"
DEFAULT_DEVICE: str ="cpu"
DEFAULT_GPU_BATCH_SIZE: int = 32

# ── Qdrant / Vector store ─────────────────────────────────────────────
DEFAULT_QDRANT_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "qdrant_store"
)
DEFAULT_COLLECTION_NAME: str = "docs"
DEFAULT_UPSERT_BATCH_SIZE: int = 100

# ── Chunking ──────────────────────────────────────────────────────────
DEFAULT_PROSE_CHUNK_SIZE: int = 350
DEFAULT_PROSE_CHUNK_OVERLAP: int = 40
