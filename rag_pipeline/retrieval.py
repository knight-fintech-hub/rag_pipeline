from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SparseVector


DEFAULT_QDRANT_PATH = "qdrant_store"
DEFAULT_COLLECTION_NAME = "docs"
DEFAULT_TOP_K = 8
DEFAULT_SCORE_THRESHOLD = 0.0
RRF_K = 60


class Retrieval:

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.top_k = top_k
        self.score_threshold = score_threshold

    # ── Dense search ──────────────────────────────────────────────────────────

    def dense_search(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        payload_filter: Optional[Filter] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        target_col = collection_name or self.collection_name
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=target_col,
                query=query_vector,
                using="dense",
                limit=k,
                score_threshold=self.score_threshold,
                query_filter=payload_filter,
                with_payload=True,
            )
            hits = res.points
        else:
            hits = self.client.search(
                collection_name=target_col,
                query_vector=query_vector,
                using="dense",
                limit=k,
                score_threshold=self.score_threshold,
                query_filter=payload_filter,
                with_payload=True,
            )
        results = []
        for hit in hits:
            results.append({
                "id": str(hit.id),
                "score": hit.score,
                "page_content": (hit.payload or {}).get("page_content", ""),
                "metadata": (hit.payload or {}).get("metadata", {}),
            })
        print(f"[Retrieval] Dense search returned {len(results)} results from '{target_col}'.")
        return results

    # ── BM25 sparse search ────────────────────────────────────────────────────

    def bm25_search(
        self,
        query_sparse_vector: Tuple[List[int], List[float]],
        top_k: Optional[int] = None,
        payload_filter: Optional[Filter] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the 'sparse-bm25' vector slot in Qdrant using the sparse vector
        produced by BGE-M3's SPLADE head. Returns keyword-matched chunks.

        Args:
            query_sparse_vector: (indices, values) tuple from Embedding.embed_query_sparse()
        """
        k = top_k or self.top_k
        target_col = collection_name or self.collection_name
        indices, values = query_sparse_vector

        if not indices:
            print("[Retrieval] BM25 query vector is empty — skipping sparse search.")
            return []

        sparse_vec = SparseVector(indices=indices, values=values)

        try:
            if hasattr(self.client, "query_points"):
                res = self.client.query_points(
                    collection_name=target_col,
                    query=sparse_vec,
                    using="sparse-bm25",
                    limit=k,
                    query_filter=payload_filter,
                    with_payload=True,
                )
                hits = res.points
            else:
                # Older qdrant-client fallback
                hits = self.client.search(
                    collection_name=target_col,
                    query_vector=("sparse-bm25", sparse_vec),
                    limit=k,
                    query_filter=payload_filter,
                    with_payload=True,
                )
        except Exception as e:
            # Graceful fallback: if sparse slot has no data yet, skip it
            print(f"[Retrieval] BM25 search failed (sparse slot may be empty): {e}")
            return []

        results = []
        for hit in hits:
            results.append({
                "id": str(hit.id),
                "score": hit.score,
                "page_content": (hit.payload or {}).get("page_content", ""),
                "metadata": (hit.payload or {}).get("metadata", {}),
            })
        print(f"[Retrieval] BM25 search returned {len(results)} results from '{target_col}'.")
        return results

    # ── MMR search (kept for optional use) ───────────────────────────────────

    def mmr_search(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        payload_filter: Optional[Filter] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        target_col = collection_name or self.collection_name
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=target_col,
                query=query_vector,
                using="dense",
                limit=fetch_k,
                query_filter=payload_filter,
                with_payload=True,
                with_vectors=True,
            )
            candidates_hits = res.points
        else:
            candidates_hits = self.client.search(
                collection_name=target_col,
                query_vector=query_vector,
                using="dense",
                limit=fetch_k,
                query_filter=payload_filter,
                with_payload=True,
                with_vectors=True,
            )

        if not candidates_hits:
            return []

        candidates = []
        for hit in candidates_hits:
            vec = hit.vector.get("dense") if isinstance(hit.vector, dict) else hit.vector
            candidates.append({
                "id": str(hit.id),
                "score": hit.score,
                "vector": np.array(vec) if isinstance(vec, list) else np.array([]),
                "page_content": (hit.payload or {}).get("page_content", ""),
                "metadata": (hit.payload or {}).get("metadata", {}),
            })

        selected: List[Dict[str, Any]] = []
        remaining = candidates[:]

        while len(selected) < k and remaining:
            if not selected:
                best = max(remaining, key=lambda x: x["score"])
            else:
                sel_vecs = np.stack([s["vector"] for s in selected])  # (n_sel, dim)

                def mmr_score(candidate: Dict) -> float:
                    rel = candidate["score"]
                    cand_vec = candidate["vector"]
                    if cand_vec.size == 0 or sel_vecs.shape[1] != cand_vec.shape[0]:
                        return rel
                    # Vectorised cosine similarity against all selected
                    norms = np.linalg.norm(sel_vecs, axis=1) * np.linalg.norm(cand_vec)
                    sims = np.dot(sel_vecs, cand_vec) / np.where(norms == 0, 1e-10, norms)
                    max_sim = float(sims.max())
                    return lambda_mult * rel - (1 - lambda_mult) * max_sim

                best = max(remaining, key=mmr_score)

            selected.append(best)
            remaining.remove(best)

        for doc in selected:
            doc.pop("vector", None)

        print(f"[Retrieval] MMR search returned {len(selected)} results from '{target_col}'.")
        return selected

    # ── RRF fusion ────────────────────────────────────────────────────────────

    def rrf_fuse(
        self,
        rankings: List[List[str]],
        k: int = RRF_K,
    ) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: -x[1])

    # ── True hybrid search: Dense + BM25 + RRF ───────────────────────────────

    def hybrid_search(
        self,
        query_vector: List[float],
        query_sparse_vector: Optional[Tuple[List[int], List[float]]] = None,
        top_k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        payload_filter: Optional[Filter] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        True hybrid search: Dense semantic search + BM25 keyword search fused via RRF.

        - Dense  → finds semantically similar chunks (paraphrases, concepts)
        - BM25   → finds exact keyword matches (codes, names, section numbers)
        - RRF    → combines both ranked lists into one final ranking

        If query_sparse_vector is None (e.g. sparse not yet indexed), falls back
        to dense-only search gracefully.
        """
        k = top_k or self.top_k
        target_col = collection_name or self.collection_name

        # ── 1. Dense search ──────────────────────────────────────────────────
        dense_results = self.dense_search(
            query_vector,
            top_k=k,
            payload_filter=payload_filter,
            collection_name=target_col,
        )

        # ── 2. BM25 sparse search ────────────────────────────────────────────
        bm25_results: List[Dict[str, Any]] = []
        if query_sparse_vector is not None:
            bm25_results = self.bm25_search(
                query_sparse_vector,
                top_k=k,
                payload_filter=payload_filter,
                collection_name=target_col,
            )

        # ── 3. If BM25 returned nothing, just return dense results ───────────
        if not bm25_results:
            print(f"[Retrieval] BM25 unavailable — returning dense results only ({len(dense_results)} candidates).")
            for doc in dense_results:
                doc["rrf_score"] = 0.0
            return dense_results

        # ── 4. Build doc map and ranked lists for RRF ────────────────────────
        doc_map: Dict[str, Dict[str, Any]] = {}

        dense_ranking: List[str] = []
        for doc in dense_results:
            doc_map[doc["id"]] = doc
            dense_ranking.append(doc["id"])

        bm25_ranking: List[str] = []
        for doc in bm25_results:
            doc_map[doc["id"]] = doc
            bm25_ranking.append(doc["id"])

        # ── 5. RRF fusion ─────────────────────────────────────────────────────
        fused = self.rrf_fuse([dense_ranking, bm25_ranking])

        seen = set()
        merged: List[Dict[str, Any]] = []
        for doc_id, rrf_score in fused:
            if doc_id not in seen and doc_id in doc_map:
                seen.add(doc_id)
                result = dict(doc_map[doc_id])
                result["rrf_score"] = rrf_score
                merged.append(result)

        print(f"[Retrieval] Hybrid (Dense+BM25 RRF) returned {len(merged)} candidates from '{target_col}'.")
        return merged

    # ── Public entry point ────────────────────────────────────────────────────

    def retrieve(
        self,
        query_vector: List[float],
        query_sparse_vector: Optional[Tuple[List[int], List[float]]] = None,
        top_k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        payload_filter: Optional[Filter] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.hybrid_search(
            query_vector=query_vector,
            query_sparse_vector=query_sparse_vector,
            top_k=top_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            payload_filter=payload_filter,
            collection_name=collection_name,
        )

    def __repr__(self) -> str:
        return (
            f"Retrieval("
            f"collection='{self.collection_name}', "
            f"top_k={self.top_k}, "
            f"score_threshold={self.score_threshold})"
        )


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Kept for backwards compatibility. numpy-backed for performance."""
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)