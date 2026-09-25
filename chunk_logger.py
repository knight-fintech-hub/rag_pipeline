
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List

from logger import get_logger

logger = get_logger(__name__)

# chunk_logger.py lives at the project root (e:\Riya Rajput\rag\),
# so one dirname() is all we need.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # → e:\Riya Rajput\rag
CHUNKS_DIR = os.path.join(_BASE_DIR, "session_chunks")   # → e:\Riya Rajput\rag\session_chunks


def _session_file(file_key: str) -> str:
    """Return the absolute path to the JSONL log file for the given key."""
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    # Sanitise so it's safe to use as a filename
    safe_key = "".join(c if (c.isalnum() or c in "-_") else "_" for c in file_key)
    return os.path.join(CHUNKS_DIR, f"{safe_key}.jsonl")

def log_chunks(
    session_id: str,
    query: str,
    reranked_docs: List[Dict[str, Any]],
    mode: str = "rag",
) -> None:
    """
    Append one turn entry to a JSONL file containing the top reranked
    chunks with their full content and scores.

    Always writes when called — the caller is responsible for checking
    the DEBUG flag before invoking this function.

    Parameters
    ----------
    session_id   : DB session UUID when use_db=True, otherwise empty string.
                   When empty a timestamped filename is used so the file is
                   still written in stateless / use_db=False mode.
    query        : The user's question for this turn.
    reranked_docs: Output of Reranker.rerank() — already sorted best-first.
    mode         : "rag" | "direct" (direct = no chunks, still writes entry).
    """
    # When there is no DB session use a timestamp-based key so we still
    # get a file on disk even in stateless / use_db=False mode.
    file_key = session_id if session_id else datetime.now().strftime("no_session_%Y%m%d_%H%M%S_%f")
    path = _session_file(file_key)

    # ── Build chunk entries ───────────────────────────────────────────────────
    chunk_entries = []
    for rank, doc in enumerate(reranked_docs, start=1):
        chunk_entries.append({
            "rank":         rank,
            "rerank_score": round(float(doc.get("rerank_score", 0.0)), 6),
            "page_content": doc.get("page_content", ""),
            "metadata":     doc.get("metadata", {}),
        })

    # ── Calculate turn number ────────────────────────────────────────────────
    # For JSONL, we can either count lines or just omit the turn number. 
    # Counting lines requires reading, so we omit 'turn' or set it to timestamp.
    
    # ── Append new turn ───────────────────────────────────────────────────────
    turn_entry = {
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,   # empty string when use_db=False / stateless
        "query":      query,
        "mode":       mode,
        "chunks":     chunk_entries,
    }

    # ── Write back ────────────────────────────────────────────────────────────
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn_entry, ensure_ascii=False) + "\n")
        logger.debug(
            f"[ChunkLogger] Logged {len(chunk_entries)} chunk(s) for "
            f"session '{session_id or '(none)'}' → {path}"
        )
    except Exception as e:
        logger.warning(f"[ChunkLogger] Could not write {path}: {e}")
