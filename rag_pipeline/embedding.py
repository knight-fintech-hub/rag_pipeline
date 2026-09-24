from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = r"E:\ds-team\huggingface_models\Old BKP\test_models\bge_m3"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
            self._model = SentenceTransformer(resolved, device=self.device)
        return self._model

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
        print(f"[Embedding] Done — {len(embeddings)} vectors produced.")
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

    def __repr__(self) -> str:
        return (
            f"Embedding("
            f"model='{os.path.basename(self.embedding_model_path)}', "
            f"device='{self.device}', "
            f"gpu_batch_size={self.gpu_batch_size})"
        )