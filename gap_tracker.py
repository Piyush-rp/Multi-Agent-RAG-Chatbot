"""Persists every low-confidence query so a human can review what the
knowledge base is missing -- this is the piece that turns a passive RAG
chatbot into an active knowledge-maintenance tool."""
import sqlite3
import os
import datetime
from config import Config


def _connect():
    os.makedirs(os.path.dirname(Config.GAP_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(Config.GAP_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gaps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            confidence REAL NOT NULL,
            best_answer TEXT,
            supplemented_answer TEXT,
            resolved INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def log_gap(query: str, confidence: float, best_answer: str,
            supplemented_answer: str = "") -> int:
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO gaps (query, confidence, best_answer,
           supplemented_answer, created_at) VALUES (?, ?, ?, ?, ?)""",
        (query, confidence, best_answer, supplemented_answer,
         datetime.datetime.utcnow().isoformat()),
    )
    conn.commit()
    gap_id = cur.lastrowid
    conn.close()
    return gap_id


def set_supplemented_answer(gap_id: int, supplemented_answer: str):
    conn = _connect()
    conn.execute("UPDATE gaps SET supplemented_answer = ? WHERE id = ?",
                 (supplemented_answer, gap_id))
    conn.commit()
    conn.close()


def list_gaps(include_resolved: bool = False):
    conn = _connect()
    query = "SELECT id, query, confidence, best_answer, supplemented_answer, resolved, created_at FROM gaps"
    if not include_resolved:
        query += " WHERE resolved = 0"
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query).fetchall()
    conn.close()
    cols = ["id", "query", "confidence", "best_answer",
            "supplemented_answer", "resolved", "created_at"]
    return [dict(zip(cols, row)) for row in rows]


def mark_resolved(gap_id: int):
    conn = _connect()
    conn.execute("UPDATE gaps SET resolved = 1 WHERE id = ?", (gap_id,))
    conn.commit()
    conn.close()


def gap_stats():
    conn = _connect()
    total = conn.execute("SELECT COUNT(*) FROM gaps").fetchone()[0]
    unresolved = conn.execute(
        "SELECT COUNT(*) FROM gaps WHERE resolved = 0"
    ).fetchone()[0]
    conn.close()
    return {"total": total, "unresolved": unresolved,
            "resolved": total - unresolved}
