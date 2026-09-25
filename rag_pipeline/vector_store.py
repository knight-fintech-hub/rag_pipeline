from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    HnswConfigDiff,
    Modifier,
    PointStruct,
    ScoredPoint,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

# Suppress harmless Python interpreter teardown warnings in local Qdrant (portalocker / sys.meta_path)
try:
    from qdrant_client.local.qdrant_local import QdrantLocal
    _orig_local_close = QdrantLocal.close

    def _safe_local_close(self, *args, **kwargs):
        try:
            _orig_local_close(self, *args, **kwargs)
        except BaseException:
            pass

    QdrantLocal.close = _safe_local_close

    _orig_client_del = getattr(QdrantClient, "__del__", None)
    if _orig_client_del is not None:
        def _safe_client_del(self):
            try:
                self.close()
            except BaseException:
                pass
        QdrantClient.__del__ = _safe_client_del
except Exception:
    pass


DEFAULT_QDRANT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_store")
DEFAULT_COLLECTION_NAME = "company_docs"
DEFAULT_DISTANCE_METRIC = Distance.COSINE
DEFAULT_UPSERT_BATCH_SIZE = 100


class QdrantVectorStore:

    def __init__(
        self,
        path: str = DEFAULT_QDRANT_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.path = os.path.abspath(path)
        self.default_collection = collection_name
        self._client: Optional[QdrantClient] = None
        self._connect()

    def _connect(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        try:
            self._client = QdrantClient(path=self.path)
            print(f"[VectorStore] Connected to Qdrant at '{self.path}'")
        except Exception as exc:
            raise RuntimeError(
                f"[VectorStore] Failed to connect at '{self.path}': {exc}"
            ) from exc

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._connect()
        return self._client  # type: ignore[return-value]

    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        name = collection_name or self.default_collection
        return self.client.collection_exists(collection_name=name)

    def create_collection(
        self,
        vector_size: int,
        collection_name: Optional[str] = None,
        distance: Distance = DEFAULT_DISTANCE_METRIC,
        force_recreate: bool = False,
    ) -> None:
        name = collection_name or self.default_collection
        if self.collection_exists(name):
            if force_recreate:
                print(f"[VectorStore] Dropping '{name}' (force_recreate=True).")
                self.client.delete_collection(collection_name=name)
            else:
                print(f"[VectorStore] Collection '{name}' already exists — skipping.")
                return
        self.client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": VectorParams(
                    size=vector_size,
                    distance=distance,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=200, on_disk=False),
                )
            },
            sparse_vectors_config={
                "sparse-bm25": SparseVectorParams(
                    index=SparseIndexParams(on_disk=True),
                    modifier=Modifier.IDF,
                )
            },
        )
        print(f"[VectorStore] Created '{name}' (dense dim={vector_size}, metric={distance}, sparse=sparse-bm25).")

    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        name = collection_name or self.default_collection
        if not self.collection_exists(name):
            print(f"[VectorStore] '{name}' does not exist — nothing to delete.")
            return
        self.client.delete_collection(collection_name=name)
        print(f"[VectorStore] Deleted '{name}'.")

    def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        name = collection_name or self.default_collection
        info = self.client.get_collection(collection_name=name)
        vectors_cfg = info.config.params.vectors
        dense_cfg = vectors_cfg.get("dense") if isinstance(vectors_cfg, dict) else vectors_cfg
        return {
            "name": name,
            "status": str(info.status),
            "vectors_count": getattr(info, "vectors_count", getattr(info, "points_count", None)),
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
            "points_count": getattr(info, "points_count", None),
            "segments_count": getattr(info, "segments_count", None),
            "config": {
                "size": getattr(dense_cfg, "size", None) if dense_cfg else None,
                "distance": str(getattr(dense_cfg, "distance", "")) if dense_cfg else None,
            },
        }

    def build_points(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[PointStruct]:
        if len(texts) != len(embeddings):
            raise ValueError(
                f"[VectorStore] texts ({len(texts)}) and embeddings ({len(embeddings)}) length mismatch."
            )
        _metadatas = metadatas or [{} for _ in texts]
        return [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": embeddings[i]},
                payload={
                    "page_content": texts[i],
                    "header": _metadatas[i].get("header", ""),
                    "chunk_type": _metadatas[i].get("type", "prose"),
                    "source": _metadatas[i].get("source", ""),
                    "metadata": _metadatas[i],
                },
            )
            for i in range(len(texts))
        ]

    def upsert_points(
        self,
        points: List[PointStruct],
        collection_name: Optional[str] = None,
        batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    ) -> int:
        name = collection_name or self.default_collection
        if not points:
            print("[VectorStore] No points to upsert.")
            return 0
        total = len(points)
        print(f"[VectorStore] Upserting {total} points into '{name}' (batch={batch_size})...")
        for start in range(0, total, batch_size):
            self.client.upsert(collection_name=name, points=points[start : start + batch_size])
        print(f"[VectorStore] Done — {total} points stored in '{name}'.")
        return total

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None,
        payload_filter: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        name = collection_name or self.default_collection
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=name,
                query=query_vector,
                using="dense",
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=payload_filter,
                with_payload=True,
            )
            hits = res.points
        else:
            hits = self.client.search(
                collection_name=name,
                query_vector=query_vector,
                using="dense",
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=payload_filter,
                with_payload=True,
            )
        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "page_content": (hit.payload or {}).get("page_content", ""),
                "metadata": (hit.payload or {}).get("metadata", {}),
            }
            for hit in hits
        ]

    def __repr__(self) -> str:
        return (
            f"QdrantVectorStore("
            f"path='{self.path}', "
            f"default_collection='{self.default_collection}')"
        )