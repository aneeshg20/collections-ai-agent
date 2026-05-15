import sqlite3
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()
client = Anthropic()

conn = sqlite3.connect("collections.db")

Query1 = pd.read_sql("""
            SELECT invoices.vendor, invoices.invoice_amount, invoices.payment_term_days, agent_runs.risk_rating, agent_runs.timestamp
            FROM invoices
            LEFT JOIN agent_runs ON invoices.vendor = agent_runs.vendor""",conn
)

print(f"Query 1 Results:")
print(Query1)

Query2 = pd.read_sql("""
            SELECT invoices.vendor, invoices.invoice_amount, invoices.payment_term_days, agent_runs.risk_rating, agent_runs.timestamp
            FROM invoices
            LEFT JOIN agent_runs ON invoices.vendor = agent_runs.vendor
            WHERE agent_runs.risk_rating = "HIGH" """,conn
)

print(f"Query 2 Results:")
print(Query2)

Query3 = pd.read_sql("""
            SELECT agent_runs.risk_rating, count(agent_runs.vendor) AS count_of_invoices, SUM(invoices.invoice_amount) AS total_invoice_amount_at_risk
            FROM invoices
            LEFT JOIN agent_runs ON invoices.vendor = agent_runs.vendor
            GROUP BY agent_runs.risk_rating
            HAVING agent_runs.risk_rating = "HIGH" """,conn
)

print(f"Query 3 Results:")
print(Query3)
