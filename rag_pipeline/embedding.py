from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = r"E:\ds-team\huggingface_models\Old BKP\test_models\bge_m3"
DEFAULT_DEVICE = os.environ.get("DEVICE", "cuda")
DEFAULT_GPU_BATCH_SIZE = 32
# gpu_batch_size (explained):
# - The gpu_batch_size parameter controls the number of texts passed to the model
#   in a single encoding batch when running on a GPU. It trades off throughput
#   versus memory usage:
#     - Larger batch sizes typically increase GPU utilization and throughput (faster
#       total encoding per second) but consume proportionally more GPU memory and
#       can cause out-of-memory (OOM) errors if set too high for the model/GPU.
#     - Smaller batch sizes use less memory and reduce OOM risk, but may underutilize
#       the GPU and process inputs more slowly.
# - Default is 32; tune this value based on the model size and available GPU memory.
# - When running on CPU, use much smaller batch sizes (often 1) to avoid excessive
#   memory and CPU contention.
# #

# SparseVector is a named tuple: (indices: List[int], values: List[float])
# Used for BM25-style sparse retrieval in Qdrant.
SparseVector = Tuple[List[int], List[float]]


class Embedding:

    def __init__(
        self,
        embedding_model_path: str = DEFAULT_EMBEDDING_MODEL,
        device: str = DEFAULT_DEVICE,
        gpu_batch_size: int = DEFAULT_GPU_BATCH_SIZE,
    ) -> None:
        self.embedding_model_path = embedding_model_path
        self.device = device
        self.gpu_batch_size = gpu_batch_size
        self._model: Optional[SentenceTransformer] = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            resolved = (
                self.embedding_model_path
                if os.path.exists(self.embedding_model_path)
                else "BAAI/bge-m3"
            )
            print(f"[Embedding] Loading model from '{resolved}' on '{self.device}'...")
            try:
                # Load on CPU first to avoid meta-tensor copy errors when the
                # saved model has uninitialised (meta) weights — then move to
                # the target device explicitly so PyTorch uses to_empty() path
                # internally instead of a raw tensor copy.
                self._model = SentenceTransformer(resolved, device="cpu")
                if self.device != "cpu":
                    self._model = self._model.to(self.device)
            except (NotImplementedError, RuntimeError) as e:
                # Meta-tensor or corrupt local model — fall back to HuggingFace Hub
                print(
                    f"[Embedding] Local model load failed ({e}). "
                    "Falling back to BAAI/bge-m3 from HuggingFace Hub..."
                )
                self._model = SentenceTransformer("BAAI/bge-m3", device="cpu")
                if self.device != "cpu":
                    self._model = self._model.to(self.device)
        return self._model

    # ── Dense embeddings ──────────────────────────────────────────────────────

    def embed_chunks(self, chunks: List[Dict]) -> List[List[float]]:
        if not chunks:
            raise ValueError("[Embedding] Cannot embed an empty chunk list.")
        model = self._get_model()
        texts = [c["text"] for c in chunks]
        print(f"[Embedding] Encoding {len(texts)} chunks (batch={self.gpu_batch_size}, device={self.device})...")
        embeddings = model.encode(
            texts,
            batch_size=self.gpu_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        print(f"[Embedding] Done — {len(embeddings)} dense vectors produced.")
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, query: str) -> List[float]:
        model = self._get_model()
        vector = model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector[0].tolist()

    # ── Sparse (BM25-style) embeddings via BGE-M3 ────────────────────────────
    # BGE-M3 supports sparse output natively through its internal SPLADE head.
    # We use the FlagEmbedding library's BGEM3FlagModel for this, which gives
    # both dense and sparse vectors from one model.

    def _get_flag_model(self):
        """Lazy-load FlagEmbedding BGEM3FlagModel for sparse encoding."""
        if not hasattr(self, "_flag_model") or self._flag_model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError:
                raise ImportError(
                    "[Embedding] FlagEmbedding library not found. "
                    "Install it with: pip install FlagEmbedding"
                )
            resolved = (
                self.embedding_model_path
                if os.path.exists(self.embedding_model_path)
                else "BAAI/bge-m3"
            )
            use_fp16 = (self.device == "cuda")
            print(f"[Embedding] Loading FlagEmbedding BGEM3 from '{resolved}' (fp16={use_fp16})...")
            try:
                self._flag_model = BGEM3FlagModel(resolved, use_fp16=use_fp16)
            except (NotImplementedError, RuntimeError) as e:
                print(
                    f"[Embedding] FlagEmbedding local load failed ({e}). "
                    "Falling back to BAAI/bge-m3 from HuggingFace Hub..."
                )
                self._flag_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16)
            print("[Embedding] BGEM3FlagModel loaded.")
        return self._flag_model

    def _output_to_sparse_vector(self, sparse_output: Dict) -> SparseVector:
        """
        Convert BGEM3FlagModel sparse output dict to (indices, values) tuple.
        sparse_output format: {token_id (int): weight (float), ...}
        """
        indices = list(sparse_output.keys())
        values = [float(sparse_output[i]) for i in indices]
        return (indices, values)

    def embed_chunks_sparse(self, chunks: List[Dict]) -> List[SparseVector]:
        """
        Encode chunks into sparse (BM25-style) vectors using BGE-M3's SPLADE head.
        Returns list of (indices, values) tuples, one per chunk.
        """
        if not chunks:
            raise ValueError("[Embedding] Cannot sparse-embed an empty chunk list.")
        model = self._get_flag_model()
        texts = [c["text"] for c in chunks]
        print(f"[Embedding] Sparse-encoding {len(texts)} chunks (batch={self.gpu_batch_size})...")

        sparse_vectors: List[SparseVector] = []
        for start in range(0, len(texts), self.gpu_batch_size):
            batch = texts[start: start + self.gpu_batch_size]
            output = model.encode(
                batch,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
                batch_size=self.gpu_batch_size,
            )
            # output["lexical_weights"] is a list of dicts: [{token_id: weight, ...}, ...]
            for sparse_dict in output["lexical_weights"]:
                sparse_vectors.append(self._output_to_sparse_vector(sparse_dict))

        print(f"[Embedding] Done — {len(sparse_vectors)} sparse vectors produced.")
        return sparse_vectors

    def embed_query_sparse(self, query: str) -> SparseVector:
        """
        Encode a single query into a sparse vector using BGE-M3's SPLADE head.
        Returns (indices, values) tuple.
        """
        model = self._get_flag_model()
        output = model.encode(
            [query],
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=1,
        )
        return self._output_to_sparse_vector(output["lexical_weights"][0])

    def __repr__(self) -> str:
        return (
            f"Embedding("
            f"model='{os.path.basename(self.embedding_model_path)}', "
            f"device='{self.device}', "
            f"gpu_batch_size={self.gpu_batch_size})"
        )