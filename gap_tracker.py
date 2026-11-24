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
    # Answers users taught for gaps, after the curator agent checked them.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS learned_answers (
            gap_id INTEGER PRIMARY KEY REFERENCES gaps(id),
            question TEXT NOT NULL,
            user_input TEXT NOT NULL,
            answer TEXT NOT NULL,
            verdict TEXT NOT NULL,
            note TEXT,
            support TEXT,
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
    query = """SELECT g.id, g.query, g.confidence, g.best_answer,
                      g.supplemented_answer, g.resolved, g.created_at,
                      l.answer, l.verdict
               FROM gaps g LEFT JOIN learned_answers l ON l.gap_id = g.id"""
    if not include_resolved:
        query += " WHERE g.resolved = 0"
    query += " ORDER BY g.created_at DESC"
    rows = conn.execute(query).fetchall()
    conn.close()
    cols = ["id", "query", "confidence", "best_answer",
            "supplemented_answer", "resolved", "created_at",
            "learned_answer", "learned_verdict"]
    return [dict(zip(cols, row)) for row in rows]


def save_learned_answer(gap_id: int, question: str, user_input: str, answer: str,
                        verdict: str, note: str = "", support: str = ""):
    """Store a taught answer and mark its gap resolved."""
    conn = _connect()
    conn.execute(
        """INSERT OR REPLACE INTO learned_answers (gap_id, question, user_input,
           answer, verdict, note, support, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (gap_id, question, user_input, answer, verdict, note, support,
         datetime.datetime.utcnow().isoformat()),
    )
    conn.execute("UPDATE gaps SET resolved = 1 WHERE id = ?", (gap_id,))
    conn.commit()
    conn.close()


def forget_learned_answer(gap_id: int):
    """Remove a taught answer and reopen its gap."""
    conn = _connect()
    conn.execute("DELETE FROM learned_answers WHERE gap_id = ?", (gap_id,))
    conn.execute("UPDATE gaps SET resolved = 0 WHERE id = ?", (gap_id,))
    conn.commit()
    conn.close()


def list_learned_answers():
    conn = _connect()
    rows = conn.execute(
        "SELECT gap_id, question, answer, support FROM learned_answers ORDER BY gap_id"
    ).fetchall()
    conn.close()
    return [dict(zip(["gap_id", "question", "answer", "support"], row)) for row in rows]


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
