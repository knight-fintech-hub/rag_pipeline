"""
rag_pipeline
------------
Modular RAG pipeline components.
"""
from rag_pipeline.config import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GPU_BATCH_SIZE,
    DEFAULT_PROSE_CHUNK_OVERLAP,
    DEFAULT_PROSE_CHUNK_SIZE,
    DEFAULT_QDRANT_PATH,
    DEFAULT_UPSERT_BATCH_SIZE,
)
from rag_pipeline.embedding import Embedding
from rag_pipeline.generation import Generation
from rag_pipeline.ingestion import DocumentIngestion
from rag_pipeline.rag_agent import RAGAgent
from rag_pipeline.reranking import Reranker
from rag_pipeline.retrieval import Retrieval
from rag_pipeline.vector_store import QdrantVectorStore

__all__ = [
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_DEVICE",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_GPU_BATCH_SIZE",
    "DEFAULT_PROSE_CHUNK_OVERLAP",
    "DEFAULT_PROSE_CHUNK_SIZE",
    "DEFAULT_QDRANT_PATH",
    "DEFAULT_UPSERT_BATCH_SIZE",
    "DocumentIngestion",
    "Embedding",
    "Generation",
    "QdrantVectorStore",
    "RAGAgent",
    "Reranker",
    "Retrieval",
]
