import sqlite3
import pandas as pd

conn = sqlite3.connect("collections.db")
conn.execute("DROP TABLE IF EXISTS agent_runs")
conn.execute("""
    CREATE TABLE agent_runs (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        vendor TEXT,
        risk_rating TEXT,
        timestamp TEXT,
        tokens_used INTEGER,
        model TEXT,
        days_overdue INTEGER,
        amount_tier TEXT,
        overdue_flag INTEGER,
        reasoning TEXT
    )
""")

summary = pd.read_sql("""
            SELECT * FROM agent_runs""",conn
)

print(summary)

conn.commit()
conn.close()