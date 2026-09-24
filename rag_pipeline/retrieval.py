from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Filter


DEFAULT_QDRANT_PATH = "qdrant_store"
DEFAULT_COLLECTION_NAME = "company_docs"
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

    def dense_search(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        payload_filter: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=k,
                score_threshold=self.score_threshold,
                query_filter=payload_filter,
                with_payload=True,
            )
            hits = res.points
        else:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
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
        print(f"[Retrieval] Dense search returned {len(results)} results.")
        return results

    def mmr_search(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        payload_filter: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=fetch_k,
                query_filter=payload_filter,
                with_payload=True,
                with_vectors=True,
            )
            candidates_hits = res.points
        else:
            candidates_hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=fetch_k,
                query_filter=payload_filter,
                with_payload=True,
                with_vectors=True,
            )

        if not candidates_hits:
            return []

        candidates = []
        for hit in candidates_hits:
            candidates.append({
                "id": str(hit.id),
                "score": hit.score,
                "vector": hit.vector if isinstance(hit.vector, list) else [],
                "page_content": (hit.payload or {}).get("page_content", ""),
                "metadata": (hit.payload or {}).get("metadata", {}),
            })

        selected: List[Dict[str, Any]] = []
        remaining = candidates[:]

        while len(selected) < k and remaining:
            if not selected:
                best = max(remaining, key=lambda x: x["score"])
            else:
                def mmr_score(candidate: Dict) -> float:
                    rel = candidate["score"]
                    max_sim = max(
                        _cosine_similarity(candidate["vector"], s["vector"])
                        for s in selected
                    ) if selected else 0.0
                    return lambda_mult * rel - (1 - lambda_mult) * max_sim

                best = max(remaining, key=mmr_score)

            selected.append(best)
            remaining.remove(best)

        for doc in selected:
            doc.pop("vector", None)

        print(f"[Retrieval] MMR search returned {len(selected)} results.")
        return selected

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

    def hybrid_search(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        payload_filter: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        k = top_k or self.top_k

        dense_results = self.dense_search(query_vector, top_k=k, payload_filter=payload_filter)
        mmr_results = self.mmr_search(query_vector, top_k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, payload_filter=payload_filter)

        doc_map: Dict[str, Dict[str, Any]] = {}

        dense_ranking: List[str] = []
        for doc in dense_results:
            doc_map[doc["id"]] = doc
            dense_ranking.append(doc["id"])

        mmr_ranking: List[str] = []
        for doc in mmr_results:
            doc_map[doc["id"]] = doc
            mmr_ranking.append(doc["id"])

        fused = self.rrf_fuse([dense_ranking, mmr_ranking])

        seen = set()
        merged: List[Dict[str, Any]] = []
        for doc_id, rrf_score in fused:
            if doc_id not in seen and doc_id in doc_map:
                seen.add(doc_id)
                result = dict(doc_map[doc_id])
                result["rrf_score"] = rrf_score
                merged.append(result)

        print(f"[Retrieval] Hybrid (RRF) fusion returned {len(merged)} candidates.")
        return merged

    def retrieve(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        payload_filter: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        return self.hybrid_search(
            query_vector=query_vector,
            top_k=top_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            payload_filter=payload_filter,
        )

    def __repr__(self) -> str:
        return (
            f"Retrieval("
            f"collection='{self.collection_name}', "
            f"top_k={self.top_k}, "
            f"score_threshold={self.score_threshold})"
        )


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)