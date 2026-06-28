import sqlite3
conn = sqlite3.connect("collections.db")
try:
    conn.execute("ALTER TABLE agent_runs ADD COLUMN recommended_action TEXT")
    conn.commit()
    print("Column added successfully")
except Exception as e:
    print(f"Note: {e}")
conn.close()