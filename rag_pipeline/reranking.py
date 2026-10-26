from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from FlagEmbedding import FlagReranker


DEFAULT_RERANKER_MODEL = r"E:\ds-team\huggingface_models\Old BKP\test_models\bge_reranker_v2_m3"
DEFAULT_TOP_N = 3


class Reranker:

    def __init__(
        self,
        model_path: str = DEFAULT_RERANKER_MODEL,
        use_fp16: bool = True,
        top_n: int = DEFAULT_TOP_N,
    ) -> None:
        self.model_path = model_path
        self.top_n = top_n
        self._model: Optional[FlagReranker] = None
        self.use_fp16 = use_fp16

    def _get_model(self) -> FlagReranker:
        if self._model is None:
            resolved = (
                self.model_path
                if os.path.exists(self.model_path)
                else "BAAI/bge-reranker-v2-m3"
            )
            print(f"[Reranker] Loading model from '{resolved}'...")
            self._model = FlagReranker(resolved, use_fp16=self.use_fp16)
        return self._model

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not docs:
            return docs

        n = top_n or self.top_n
        model = self._get_model()

        # ── Deduplicate by page_content before scoring ────────────────────────
        # Duplicate chunks (same text, different vector IDs) inflate rerank
        # results and destroy Contextual Relevancy scores in evaluation.
        seen_content: set = set()
        unique_docs: List[Dict[str, Any]] = []
        for doc in docs:
            content = doc.get("page_content", "").strip()
            if content and content not in seen_content:
                seen_content.add(content)
                unique_docs.append(doc)

        if len(unique_docs) < len(docs):
            print(f"[Reranker] Deduped {len(docs)} → {len(unique_docs)} unique chunks.")

        pairs = [[query, doc["page_content"]] for doc in unique_docs]
        scores = model.compute_score(pairs, normalize=True)

        if not isinstance(scores, list):
            scores = [scores]

        ranked = sorted(zip(unique_docs, scores), key=lambda x: -x[1])

        print(f"[Reranker] Top-{n} results after reranking:")
        for doc, score in ranked[:n]:
            print(f"  score={score:.4f} | [{doc.get('metadata', {}).get('source', '')[-30:]} > {doc.get('metadata', {}).get('header', '')}]\n  {doc['page_content'][:80]}")

        results = []
        for doc, score in ranked[:n]:
            result = dict(doc)
            result["rerank_score"] = score
            results.append(result)

        return results

    def __repr__(self) -> str:
        return (
            f"Reranker("
            f"model='{os.path.basename(self.model_path)}', "
            f"top_n={self.top_n}, "
            f"use_fp16={self.use_fp16})"
        )
