from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from qdrant_client.models import Filter, FieldCondition, MatchAny

from rag_pipeline.ingestion import DocumentIngestion
from rag_pipeline.embedding import Embedding
from rag_pipeline.vector_store import QdrantVectorStore
from rag_pipeline.retrieval import Retrieval
from rag_pipeline.reranking import Reranker
from rag_pipeline.generation import Generation

# Session-aware chat history (loaded lazily so the DB is optional)
try:
    import sys, os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from config import cfg as _cfg
    from session_manager import load_history, save_turn
    from chunk_logger import log_chunks
    _SESSION_MANAGER_AVAILABLE = True
except Exception:
    _cfg = None  # type: ignore[assignment]
    _SESSION_MANAGER_AVAILABLE = False
    load_history = lambda *a, **kw: []    # type: ignore[assignment]
    save_turn    = lambda *a, **kw: None  # type: ignore[assignment]
    log_chunks   = lambda *a, **kw: None  # type: ignore[assignment]


def _use_db() -> bool:
    """Return True when the DB feature flag is enabled and the session manager loaded."""
    return _SESSION_MANAGER_AVAILABLE and (_cfg is not None) and _cfg.USE_DB


def _debug() -> bool:
    """Return True when debug chunk logging is enabled."""
    return _SESSION_MANAGER_AVAILABLE and (_cfg is not None) and _cfg.DEBUG


def _print_chunks(
    question: str,
    docs: List[Dict[str, Any]],
    mode: str = "rag",
) -> None:
    """Print retrieved/reranked chunks to the terminal when DEBUG=true.

    Works independently of session_id — call it for every query.
    """
    sep = "─" * 60
    print(f"\n[DEBUG] {sep}")
    print(f"[DEBUG] Query : {question}")
    print(f"[DEBUG] Mode  : {mode}  |  Chunks: {len(docs)}")
    print(f"[DEBUG] {sep}")
    if not docs:
        print("[DEBUG]  (no chunks retrieved)")
    for i, doc in enumerate(docs, start=1):
        score = doc.get("rerank_score", doc.get("score", 0.0))
        meta  = doc.get("metadata", {})
        text  = doc.get("page_content", "")
        preview = text[:300].replace("\n", " ") + ("…" if len(text) > 300 else "")
        print(f"[DEBUG]  [{i}] score={score:.4f}  meta={meta}")
        print(f"[DEBUG]       {preview}")
    print(f"[DEBUG] {sep}\n")





DEFAULT_EMBEDDING_MODEL  = r"E:\ds-team\huggingface_models\Old BKP\test_models\bge_m3"
DEFAULT_RERANKER_MODEL   = r"E:\ds-team\huggingface_models\Old BKP\test_models\bge_reranker_v2_m3"
DEFAULT_GENERATOR_MODEL  = r"E:\DS_Team_Share\hugging-face-model\Qwen2.5-14B-Instruct\models--Qwen--Qwen2.5-14B-Instruct\snapshots\cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
DEFAULT_QDRANT_PATH      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_store")
DEFAULT_COLLECTION_NAME  = "docs"
DEFAULT_DEVICE           = "cuda"
DEFAULT_TOP_K            = 8
DEFAULT_TOP_N            = 3
DEFAULT_SCORE_THRESHOLD  = 0.4


class RAGAgent:

    def __init__(
        self,
        pdf_paths: List[str] = None,
        embedding_model_path: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model_path: str = DEFAULT_RERANKER_MODEL,
        generator_model_path: str = DEFAULT_GENERATOR_MODEL,
        qdrant_path: str = DEFAULT_QDRANT_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        device: str = DEFAULT_DEVICE,
        top_k: int = DEFAULT_TOP_K,
        top_n: int = DEFAULT_TOP_N,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        gpu_batch_size: int = 32,
        max_new_tokens: int = 512,
        quantize: bool = True,
        force_recreate= False
        
    ) -> None:
        self.collection_name = collection_name  
        self.score_threshold = score_threshold
        
        self.documents = pdf_paths
        self.force_recreate= force_recreate

        self.ingestion = DocumentIngestion()

        self.embedder = Embedding(
            embedding_model_path=embedding_model_path,
            device=device,
            gpu_batch_size=gpu_batch_size,
        )

        self.vector_store = QdrantVectorStore(
            path=qdrant_path,
            collection_name=collection_name,
        )

        self.retriever = Retrieval(
            client=self.vector_store.client,
            collection_name=collection_name,
            top_k=top_k,
        )

        self.reranker = Reranker(
            model_path=reranker_model_path,
            top_n=top_n,
        )

        self.llm = Generation(
            model_path=generator_model_path,
            max_new_tokens=max_new_tokens,
            quantize=quantize,
        )

    def ingest(
        self,
        pdf_paths: List[str] = None,
        force_recreate: bool =None,
        doc_ids: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        latencies = {}
        
        if pdf_paths is None:
            pdf_paths = self.documents
        if force_recreate is None:
            force_recreate = self.force_recreate

        print(f"\n[RAGAgent] Ingesting {len(pdf_paths)} document(s)...")

        t0 = time.perf_counter()
        chunks = self.ingestion.extract_chunks(pdf_paths, doc_ids=doc_ids)
        latencies["chunking"] = round(time.perf_counter() - t0, 4)
        
        if not chunks:
            raise ValueError("[RAGAgent] No chunks extracted from provided PDFs.")

        t0 = time.perf_counter()
        embeddings = self.embedder.embed_chunks(chunks)
        latencies["embedding"] = round(time.perf_counter() - t0, 4)

        t0 = time.perf_counter()
        vector_size = len(embeddings[0])
        self.vector_store.create_collection(
            vector_size=vector_size,
            collection_name=self.collection_name,
            force_recreate=self.force_recreate,
        )

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        points = self.vector_store.build_points(texts, embeddings, metadatas)
        upserted = self.vector_store.upsert_points(points, collection_name=self.collection_name)
        latencies["indexing"] = round(time.perf_counter() - t0, 4)

        latencies["total_ingestion"] = round(sum(latencies.values()), 4)
        print(f"[RAGAgent] Ingestion complete — {upserted} points indexed.")
        
        return {
            "status": "success",
            "documents": len(pdf_paths),
            "chunks": len(chunks),
            "points_indexed": upserted,
            "collection": self.collection_name,
            "latencies": latencies,
        }

    def invoke(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        t_total = time.perf_counter()

        latencies = {}
        print(f"\n[RAGAgent] ── Query: '{question}' (session={session_id or 'stateless'}) ──")

        # ── 1. Load chat history for this session ──────────────────────────
        chat_history = []
        if session_id and _use_db():
            chat_history = load_history(session_id)

        t0 = time.perf_counter()
        query_vector = self.embedder.embed_query(question)
        latencies["query_embedding"] = round(time.perf_counter() - t0, 4)
        
        payload_filter = None
        if doc_ids:
            payload_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.doc_id",
                        match=MatchAny(any=doc_ids)
                    )
                ]
            )

        t0 = time.perf_counter()
        candidates = self.retriever.retrieve(query_vector, payload_filter=payload_filter)
        latencies["retrieval"] = round(time.perf_counter() - t0, 4)

        if not candidates:
            print("[RAGAgent] No candidates retrieved — answering directly.")
            t0 = time.perf_counter()
            answer = self.llm.generate_direct(question, chat_history=chat_history or None)
            latencies["generation"] = round(time.perf_counter() - t0, 4)
            latencies["total_query"] = round(time.perf_counter() - t_total, 4)

            # ── Log chunks (empty — no retrieval) ──────────────────────────
            if _debug():
                _print_chunks(question, [], mode="direct")
                log_chunks(session_id or "", question, [], mode="direct")

            # ── Save turn ──────────────────────────────────────────────────
            if session_id and _use_db():
                save_turn(session_id, question, answer)

            return {
                "answer": answer,
                "mode": "direct",
                "reason": "no_candidates",
                "context": "",
                "docs": [],
                "query": question,
                "latencies": latencies,
            }

        t0 = time.perf_counter()
        reranked = self.reranker.rerank(question, candidates)
        latencies["reranking"] = round(time.perf_counter() - t0, 4)

        top_score = reranked[0].get("rerank_score", 0.0) if reranked else 0.0
        print(f"[RAGAgent] Top rerank score: {top_score:.4f} (threshold={self.score_threshold})")

        # ── Log reranked chunks once (shared by both paths below) ──────────
        if _debug():
            _mode = "rag" if top_score >= self.score_threshold else "direct_low_score"
            _print_chunks(question, reranked, mode=_mode)
            log_chunks(session_id or "", question, reranked, mode=_mode)

        if top_score >= self.score_threshold:
            print("[RAGAgent] Relevant docs found — generating with RAG.")
            t0 = time.perf_counter()
            result = self.llm.generate(question, reranked, chat_history=chat_history or None)
            latencies["generation"] = round(time.perf_counter() - t0, 4)
            latencies["total_query"] = round(time.perf_counter() - t_total, 4)

            result["mode"] = "rag"
            result["docs"] = reranked
            result["top_rerank_score"] = top_score
            result["latencies"] = latencies

            # ── Save turn ──────────────────────────────────────────────────
            if session_id and _use_db():
                save_turn(session_id, question, result["answer"])

            return result

        print("[RAGAgent] Docs below threshold — answering directly.")
        t0 = time.perf_counter()
        answer = self.llm.generate_direct(question, chat_history=chat_history or None)
        latencies["generation"] = round(time.perf_counter() - t0, 4)
        latencies["total_query"] = round(time.perf_counter() - t_total, 4)

        # ── Save turn ──────────────────────────────────────────────────────
        if session_id and _use_db():
            save_turn(session_id, question, answer)

        return {
            "answer": answer,
            "mode": "direct",
            "reason": "low_score",
            "top_rerank_score": top_score,
            "context": "",
            "docs": reranked,
            "query": question,
            "latencies": latencies,
        }



    def __repr__(self) -> str:
        return (
            f"RAGAgent("
            f"collection='{self.collection_name}', "
            f"score_threshold={self.score_threshold})"
        )


if __name__ == "__main__":
    ragobj = RAGAgent()
    ans = ragobj.invoke("What is machine learning`")