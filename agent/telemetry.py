import sqlite3
import os
from datetime import datetime

DB_PATH = os.getenv("SQLITE_DB_PATH", "./db/telemetry.db")


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            invoice_id  TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            event_type  TEXT NOT NULL,   -- 'classification' | 'email_draft'
            risk_tier   TEXT,
            input_tokens  INTEGER,
            output_tokens INTEGER,
            latency_ms  INTEGER,
            notes       TEXT
        )
    """)
    conn.commit()
    return conn


def log_event(
    invoice_id: str,
    customer_id: str,
    event_type: str,
    risk_tier: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: int = 0,
    notes: str = "",
) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT INTO agent_runs
           (timestamp, invoice_id, customer_id, event_type, risk_tier,
            input_tokens, output_tokens, latency_ms, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.utcnow().isoformat(), invoice_id, customer_id,
         event_type, risk_tier, input_tokens, output_tokens, latency_ms, notes),
    )
    conn.commit()
    conn.close()


def fetch_run_summary() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("""
        SELECT event_type, risk_tier, COUNT(*) as count,
               SUM(input_tokens + output_tokens) as total_tokens,
               AVG(latency_ms) as avg_latency_ms
        FROM agent_runs
        GROUP BY event_type, risk_tier
    """).fetchall()
    conn.close()
    return [
        {"event_type": r[0], "risk_tier": r[1], "count": r[2],
         "total_tokens": r[3], "avg_latency_ms": round(r[4] or 0, 1)}
        for r in rows
    ]
