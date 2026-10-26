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

# Suppress harmless Python interpreter teardown warnings in local Qdrant
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
DEFAULT_COLLECTION_NAME = "docs"
DEFAULT_DISTANCE_METRIC = Distance.COSINE
DEFAULT_UPSERT_BATCH_SIZE = 100

# ── NOTE: Qdrant env vars are read lazily inside _connect() ──────────────────
# Reading them at module import time caused them to be evaluated BEFORE
# dotenv (config.py) had a chance to load the .env file.
# _connect() is called from __init__, which runs after dotenv is loaded.

def _qdrant_mode()  -> str:  return os.environ.get("QDRANT_MODE",    "local").strip().lower()
def _qdrant_host()  -> str:  return os.environ.get("QDRANT_HOST",    "localhost")
def _qdrant_port()  -> int:  return int(os.environ.get("QDRANT_PORT", "6333"))
def _qdrant_url()   -> str:  return os.environ.get("QDRANT_URL",     "")
def _qdrant_key()   -> str:  return os.environ.get("QDRANT_API_KEY", "")


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
        """
        Connect to Qdrant based on QDRANT_MODE env variable.

        Modes:
          local  — in-process disk storage (QdrantClient(path=...))
          server — local Qdrant server via HTTP (QdrantClient(host=, port=))
          cloud  — Qdrant Cloud (QdrantClient(url=, api_key=))
        """
        mode    = _qdrant_mode()
        host    = _qdrant_host()
        port    = _qdrant_port()
        url     = _qdrant_url()
        api_key = _qdrant_key()

        try:
            if mode == "server":
                self._client = QdrantClient(host=host, port=port)
                print(f"[VectorStore] Connected to Qdrant server at '{host}:{port}'")

            elif mode == "cloud":
                if not url:
                    raise ValueError(
                        "[VectorStore] QDRANT_MODE=cloud but QDRANT_URL is not set. "
                        "Set QDRANT_URL in your .env file."
                    )
                self._client = QdrantClient(url=url, api_key=api_key or None)
                print(f"[VectorStore] Connected to Qdrant Cloud at '{url}'")

            else:
                # Default: local disk mode
                if mode not in ("local", ""):
                    print(
                        f"[VectorStore] Warning: unknown QDRANT_MODE='{mode}', "
                        f"falling back to local disk mode."
                    )
                os.makedirs(self.path, exist_ok=True)
                self._client = QdrantClient(path=self.path)
                print(f"[VectorStore] Connected to Qdrant (local disk) at '{self.path}'")

        except Exception as exc:
            raise RuntimeError(
                f"[VectorStore] Failed to connect (mode={mode}): {exc}"
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
        sparse_vectors: Optional[List[Any]] = None,
    ) -> List[PointStruct]:
        """
        Build Qdrant PointStructs with dense vectors and optionally sparse (BM25) vectors.

        Args:
            texts:          Raw chunk texts.
            embeddings:     Dense float vectors (one per chunk).
            metadatas:      Metadata dicts (one per chunk).
            sparse_vectors: Optional list of (indices, values) tuples from BGE-M3 SPLADE head.
                            When provided, stored in the "sparse-bm25" named vector slot.
        """
        if len(texts) != len(embeddings):
            raise ValueError(
                f"[VectorStore] texts ({len(texts)}) and embeddings ({len(embeddings)}) length mismatch."
            )
        if sparse_vectors is not None and len(sparse_vectors) != len(texts):
            raise ValueError(
                f"[VectorStore] sparse_vectors ({len(sparse_vectors)}) and texts ({len(texts)}) length mismatch."
            )

        _metadatas = metadatas or [{} for _ in texts]

        from qdrant_client.models import SparseVector as QdrantSparseVector

        points = []
        for i in range(len(texts)):
            # Always store the dense vector
            vector_dict: Dict[str, Any] = {"dense": embeddings[i]}

            # Store sparse (BM25) vector when available
            if sparse_vectors is not None:
                indices, values = sparse_vectors[i]
                if indices:  # skip empty sparse vectors
                    vector_dict["sparse-bm25"] = QdrantSparseVector(
                        indices=indices,
                        values=values,
                    )

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector_dict,
                    payload={
                        "page_content": texts[i],
                        "header": _metadatas[i].get("header", ""),
                        "chunk_type": _metadatas[i].get("type", "prose"),
                        "source": _metadatas[i].get("source", ""),
                        "metadata": _metadatas[i],
                    },
                )
            )

        has_sparse = sparse_vectors is not None
        print(f"[VectorStore] Built {len(points)} points (dense=yes, sparse-bm25={has_sparse}).")
        return points

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

    def list_documents(self, collection_name: Optional[str] = None) -> List[Dict[str, str]]:
        """Return a deduplicated list of {doc_id, filename} for every ingested document."""
        name = collection_name or self.default_collection
        if not self.collection_exists(name):
            return []

        seen: dict[str, str] = {}  # doc_id -> filename
        offset = None

        while True:
            result = self.client.scroll(
                collection_name=name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = result

            for point in points:
                meta = (point.payload or {}).get("metadata", {})
                doc_id = meta.get("doc_id")
                source = meta.get("source", "")
                filename = os.path.basename(source) if source else "unknown"
                if doc_id and doc_id not in seen:
                    seen[doc_id] = filename

            if next_offset is None:
                break
            offset = next_offset

        return [{"doc_id": doc_id, "filename": filename} for doc_id, filename in seen.items()]

    def delete_document(self, doc_id: str, collection_name: Optional[str] = None) -> None:
        name = collection_name or self.default_collection
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        self.client.delete(
            collection_name=name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.doc_id",
                        match=MatchValue(value=doc_id)
                    )
                ]
            )
        )
        print(f"[VectorStore] Deleted document {doc_id} from '{name}'.")

    def __repr__(self) -> str:
        return (
            f"QdrantVectorStore("
            f"path='{self.path}', "
            f"default_collection='{self.default_collection}')"
        )