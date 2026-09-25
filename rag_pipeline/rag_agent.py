from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from qdrant_client.models import FieldCondition, Filter, MatchAny

from rag_pipeline.config import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GPU_BATCH_SIZE,
    DEFAULT_QDRANT_PATH,
)
from rag_pipeline.embedding import Embedding
from rag_pipeline.generation import DEFAULT_MODEL_PATH, Generation
from rag_pipeline.ingestion import DocumentIngestion
from rag_pipeline.reranking import DEFAULT_RERANKER_MODEL, Reranker
from rag_pipeline.retrieval import Retrieval
from rag_pipeline.vector_store import QdrantVectorStore


class RAGAgent:
    """
    RAG Agent Orchestrator
    ----------------------
    Coordinates document ingestion, chunking, embedding, vector store indexing,
    hybrid retrieval, cross-encoder reranking, and LLM answer generation.
    """

    def __init__(
        self,
        embedding_model_path: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model_path: str = DEFAULT_RERANKER_MODEL,
        generator_model_path: str = DEFAULT_MODEL_PATH,
        qdrant_path: str = DEFAULT_QDRANT_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        device: str = DEFAULT_DEVICE,
        top_k: int = 8,
        top_n: int = 3,
        score_threshold: float = 0.4,
        gpu_batch_size: int = DEFAULT_GPU_BATCH_SIZE,
        max_new_tokens: int = 512,
        quantize: bool = True,
    ) -> None:
        self.collection_name = collection_name
        self.top_k = top_k
        self.top_n = top_n
        self.score_threshold = score_threshold
        self.device = device

        self.ingestion = DocumentIngestion()
        self.embedding = Embedding(
            embedding_model_path=embedding_model_path,
            device=device,
            gpu_batch_size=gpu_batch_size,
        )
        self.vector_store = QdrantVectorStore(
            path=qdrant_path,
            collection_name=collection_name,
        )
        self.retrieval = Retrieval(
            client=self.vector_store.client,
            collection_name=collection_name,
            top_k=top_k,
            score_threshold=0.0,
        )
        self.reranker = Reranker(
            model_path=reranker_model_path,
            top_n=top_n,
            use_fp16=(device == "cuda"),
        )
        self.generation = Generation(
            model_path=generator_model_path,
            max_new_tokens=max_new_tokens,
            quantize=quantize,
        )

        print(f"[RAGAgent] Initialised with collection='{collection_name}', device='{device}'")

    def ingest(
        self,
        saved_paths: List[str],
        force_recreate: bool = False,
        doc_mapping: Optional[Dict[str, str]] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parses PDF files, computes embeddings, and stores vectors into Qdrant.
        """
        t0 = time.perf_counter()
        target_collection = collection_name or self.collection_name
        self.collection_name = target_collection

        # 1. Chunking
        t_chunk_start = time.perf_counter()
        chunks = self.ingestion.extract_chunks(pdf_paths=saved_paths, doc_ids=doc_mapping)
        t_chunk = time.perf_counter() - t_chunk_start

        if not chunks:
            return {
                "status": "warning: no chunks extracted",
                "documents": len(saved_paths),
                "chunks": 0,
                "points_indexed": 0,
                "collection": target_collection,
                "latencies": {
                    "chunking": round(t_chunk, 3),
                    "embedding": 0.0,
                    "indexing": 0.0,
                    "total": round(time.perf_counter() - t0, 3),
                },
            }

        # 2. Embedding
        t_embed_start = time.perf_counter()
        embeddings = self.embedding.embed_chunks(chunks)
        t_embed = time.perf_counter() - t_embed_start

        # 3. Vector indexing
        t_index_start = time.perf_counter()
        vector_size = len(embeddings[0])
        self.vector_store.create_collection(
            vector_size=vector_size,
            collection_name=target_collection,
            force_recreate=force_recreate,
        )
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        points = self.vector_store.build_points(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        points_indexed = self.vector_store.upsert_points(
            points=points,
            collection_name=target_collection,
        )
        t_index = time.perf_counter() - t_index_start
        total_latency = time.perf_counter() - t0

        return {
            "status": "success",
            "documents": len(saved_paths),
            "chunks": len(chunks),
            "points_indexed": points_indexed,
            "collection": target_collection,
            "latencies": {
                "chunking": round(t_chunk, 3),
                "embedding": round(t_embed, 3),
                "indexing": round(t_index, 3),
                "total": round(total_latency, 3),
            },
        }

    def invoke(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes query retrieval, reranking, and generation.
        """
        t0 = time.perf_counter()
        target_collection = collection_name or self.collection_name

        payload_filter: Optional[Filter] = None
        if doc_ids:
            payload_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.doc_id",
                        match=MatchAny(any=doc_ids),
                    )
                ]
            )

        # 1. Query Embedding
        t_embed_start = time.perf_counter()
        query_vector = self.embedding.embed_query(question)
        t_embed = time.perf_counter() - t_embed_start

        # 2. Retrieval (Hybrid Dense + MMR with RRF fusion)
        t_retrieve_start = time.perf_counter()
        retrieved_docs = self.retrieval.hybrid_search(
            query_vector=query_vector,
            top_k=self.top_k,
            payload_filter=payload_filter,
            collection_name=target_collection,
        )
        t_retrieve = time.perf_counter() - t_retrieve_start

        # 3. Reranking
        t_rerank_start = time.perf_counter()
        top_score: Optional[float] = None
        reranked_docs: List[Dict[str, Any]] = []

        if retrieved_docs:
            reranked_docs = self.reranker.rerank(
                query=question,
                docs=retrieved_docs,
                top_n=self.top_n,
            )
            if reranked_docs:
                top_score = float(reranked_docs[0].get("rerank_score", 0.0))
        t_rerank = time.perf_counter() - t_rerank_start

        # 4. Generation decision
        t_gen_start = time.perf_counter()
        if not reranked_docs:
            mode = "direct"
            reason = "No documents retrieved"
            answer = self.generation.generate_direct(question)
            context = ""
        elif top_score is not None and top_score < self.score_threshold:
            mode = "direct"
            reason = f"Top rerank score ({top_score:.3f}) below threshold ({self.score_threshold})"
            answer = self.generation.generate_direct(question)
            context = ""
        else:
            mode = "rag"
            reason = f"Relevant context found (top rerank score: {top_score:.3f})"
            gen_res = self.generation.generate(query=question, docs=reranked_docs)
            answer = gen_res.get("answer", "")
            context = gen_res.get("context", "")

        t_gen = time.perf_counter() - t_gen_start
        total_latency = time.perf_counter() - t0

        return {
            "answer": answer,
            "mode": mode,
            "query": question,
            "context": context,
            "top_rerank_score": top_score,
            "reason": reason,
            "latencies": {
                "embedding": round(t_embed, 3),
                "retrieval": round(t_retrieve, 3),
                "reranking": round(t_rerank, 3),
                "generation": round(t_gen, 3),
                "total": round(total_latency, 3),
            },
        }

    def __repr__(self) -> str:
        return (
            f"RAGAgent("
            f"collection='{self.collection_name}', "
            f"top_k={self.top_k}, "
            f"top_n={self.top_n}, "
            f"score_threshold={self.score_threshold})"
        )
