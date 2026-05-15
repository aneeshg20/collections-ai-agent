import sqlite3
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()
client = Anthropic()

conn = sqlite3.connect("collections.db")

conn.execute("""
    CREATE TABLE IF NOT EXISTS agent_runs (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        vendor TEXT,
        risk_rating TEXT,
        timestamp TEXT,
        tokens_used INTEGER,
        model TEXT
    )
""")

conn.commit()
conn.execute("DELETE FROM agent_runs")
conn.commit()

result = pd.read_sql("SELECT * FROM invoices", conn)

for _, inv in result.iterrows():
    message = client.messages.create(
    model = "claude-sonnet-4-5",
    max_tokens=1024,
    system = "You are an agent in a world class collections team for a managed services account in a top tier consultancy firm. Your job is to idetify the collection risk in terms of breaking promise to pay, possible delinquency, impact in DSO and working capital. You must also raise a flag if its going into a bad debt zone by checking customer payment history and trend as well. Give the response as High, Medium and Low and also your reasoning for the same",
    messages=[
        {"role":"user", "content": f" Assess collection risk for this invoice: {inv}"}
             ]
    
    )
    print(message.content[0].text)
    if ("HIGH" in message.content[0].text):
        rating = "HIGH"
    elif ("MEDIUM" in message.content[0].text):
        rating = "MEDIUM"
    elif ("LOW" in message.content[0].text):
        rating = "LOW"
    else:
        rating = "UNKNOWN"
    print("\n" + "="*60 + "\n")

    total_tokens = message.usage.input_tokens + message.usage.output_tokens
    risk_rating = rating

    conn.execute("""
    INSERT INTO agent_runs (vendor, risk_rating, timestamp, tokens_used, model)
    VALUES (?, ?, ?, ?, ?)
         """, (
             inv['vendor'],
             risk_rating,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             total_tokens,
             "claude-sonnet-4-5"
         )
    )

    conn.commit()


runs = pd.read_sql("SELECT * FROM agent_runs", conn)
print (f"\n Telemetry log:")
print (runs)

summary = pd.read_sql("""
            SELECT risk_rating, COUNT(*) as count, SUM(tokens_used) as total_tokens
            FROM agent_runs
            GROUP BY risk_rating""",conn
)
print(f"Risk Summary:")
print(summary)

conn.close()