"""
Session Manager — reads and writes chat history from the existing
`messages` table in PostgreSQL.

Existing table schema (not created here — already in the DB):

    messages (
        id          SERIAL PRIMARY KEY,
        session_id  TEXT        NOT NULL,
        role        TEXT        NOT NULL,   -- 'user' | 'assistant'
        content     TEXT        NOT NULL,
        node        TEXT,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
"""
from __future__ import annotations

from typing import List, Dict, Optional

import threading
from database import execute_query, execute_write_many
from logger import get_logger

logger = get_logger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def build_retrieval_query(
    question: str,
    history: List[Dict[str, str]],
) -> str:
    """Include prior session messages when forming the document search query."""
    if not history:
        return question

    conversation = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in history
    )
    return f"Conversation history:\n{conversation}\n\nCurrent question:\n{question}"


def load_history(
    session_id: str,
    max_turns: int = 10,
) -> List[Dict[str, str]]:
    """
    Return the last *max_turns* (user + assistant pairs) for the session,
    ordered oldest-first so they can be prepended to the LLM prompt.

    Each element is {"role": "user"|"assistant", "content": "..."}.
    """
    sql = """
        SELECT role, content
        FROM   messages
        WHERE  session_id = %s
        ORDER  BY created_at DESC
        LIMIT  %s
    """
    try:
        rows = execute_query(sql, (session_id, max_turns * 2))
        # rows come back newest-first; reverse for chronological order
        history = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
        logger.debug(
            f"[SessionManager] Loaded {len(history)} messages "
            f"for session '{session_id}'."
        )
        return history
    except Exception as e:
        logger.warning(
            f"[SessionManager] Could not load history for '{session_id}': {e}"
        )
        return []

def _save_turn_task(session_id: str, question: str, answer: str, node: Optional[str]):
    sql = """
        INSERT INTO messages (session_id, role, content, node)
        VALUES (%s, %s, %s, %s)
    """
    try:
        params_list = [
            (session_id, "user", question, node),
            (session_id, "assistant", answer, node)
        ]
        execute_write_many(sql, params_list)
        logger.debug(
            f"[SessionManager] Saved turn for session '{session_id}'."
        )
    except Exception as e:
        logger.warning(
            f"[SessionManager] Could not save turn for '{session_id}': {e}"
        )

def save_turn(
    session_id: str,
    question: str,
    answer: str,
    node: Optional[str] = "rag",
) -> None:
    """
    Append the user question and assistant answer as two rows
    in the existing `messages` table asynchronously.
    """
    threading.Thread(
        target=_save_turn_task,
        args=(session_id, question, answer, node),
        daemon=True
    ).start()
