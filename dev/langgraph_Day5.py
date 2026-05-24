from anthropic import Anthropic
from dotenv import load_dotenv
from datetime import datetime
import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import sqlite3
import pandas as pd

load_dotenv()
client = Anthropic()

class InvoiceState (TypedDict):
    vendor: str
    invoice_amount: float
    days_since_invoice: int
    payment_term_days: int
    risk_rating: str
    reasoning: str
    # New fields for pre_classify
    overdue_flag: bool
    days_overdue: int
    amount_tier: str  # "HIGH_VALUE" / "MEDIUM_VALUE" / "LOW_VALUE"

def pre_classify(state: InvoiceState):
    overdue_flag = state['days_since_invoice'] > state['payment_term_days']
    days_overdue = state['days_since_invoice'] - state['payment_term_days'] if overdue_flag else 0
    
    if state['invoice_amount'] > 1000000:
        amount_tier = "HIGH VALUE"
    elif state['invoice_amount'] > 300000:
        amount_tier = "MEDIUM VALUE"
    else:
        amount_tier = "LOW VALUE"
    
    return {
        "overdue_flag": overdue_flag,
        "days_overdue": days_overdue,
        "amount_tier": amount_tier
    }

def assess_risk(state: InvoiceState):
    message = client.messages.create(
    model = "claude-sonnet-4-5",
    max_tokens=1024,
    system = f"""You are an agent in a world class collections team for a managed services account in a top tier consultancy firm.
    Your job is to idetify the collection risk in terms of breaking promise to pay, possible delinquency, impact in DSO and working capital. You must also raise a flag if its going into a bad debt zone by checking customer payment history and trend as well. 
    Invoice details:
    Vendor: {state['vendor']}
    Amount: {state['invoice_amount']} ({state['amount_tier']})
    Days overdue: {state['days_overdue']}
    Overdue: {state['overdue_flag']}

    Pre-classification says this is a {state['amount_tier']} 
    invoice that is {'overdue' if state['overdue_flag'] else 'current'}.
    Assess collection risk as HIGH/MEDIUM/LOW with reasoning.
    """,
    messages=[
        {"role":"user", "content": f" Assess collection risk for this invoice: {state}"}
    ]
    )
    if ("HIGH" in message.content[0].text):
        rating = "HIGH"
    elif ("MEDIUM" in message.content[0].text):
        rating = "MEDIUM"
    elif ("LOW" in message.content[0].text):
        rating = "LOW"
    else:
        rating = "UNKNOWN"
    return {"risk_rating": rating, "reasoning": message.content[0].text}


def log_telemetry (state: InvoiceState):
    conn = sqlite3.connect("collections.db")
    conn.execute("""
                 INSERT INTO agent_runs(vendor, risk_rating, timestamp, tokens_used, model,days_overdue,amount_tier,overdue_flag,reasoning)
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (
                      state['vendor'],
                      state['risk_rating'],
                      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                      512,
                      "claude-sonnet-4-5",
                      state['days_overdue'],
                      state['amount_tier'],
                      state['overdue_flag'],
                      state['reasoning']
                  )
                 )
    conn.commit()
    conn.close()

    return{}

def escalate(state: InvoiceState):
    print(f"\n🚨 ESCALATION ALERT 🚨")
    print(f"Vendor: {state['vendor']}")
    print(f"Amount: ₹{state['invoice_amount']:,.0f}")
    print(f"Days Overdue: {state['days_overdue']}")
    print(f"Immediate senior management review required")
    return {}

def route_by_risk(state:InvoiceState) ->str:
    if state['risk_rating'] == "HIGH":
        return "escalate"
    else:
        return "log_telemetry"

# Build the graph
graph = StateGraph(InvoiceState)
graph.add_node("pre_classify", pre_classify)
graph.add_edge(START, "pre_classify")
graph.add_node("assess_risk", assess_risk)
graph.add_edge("pre_classify","assess_risk")
graph.add_node("escalate", escalate)
graph.add_node("log_telemetry", log_telemetry)
graph.add_conditional_edges(
    "assess_risk",        # from this node
    route_by_risk,        # use this function to decide
    {
        "escalate": "escalate",           # if returns "escalate"
        "log_telemetry": "log_telemetry"  # if returns "log_telemetry"
    }
)
graph.add_edge("escalate","log_telemetry")
graph.add_edge("log_telemetry", END)
app = graph.compile()

conn = sqlite3.connect("collections.db")

# Load invoices from CSV
data = pd.read_csv("agent/invoices.csv")

# Loop through each row
for _, row in data.iterrows():
    invoice = {
        "vendor": row['vendor'],
        "invoice_amount": float(row['invoice_amount']),
        "days_since_invoice": int(row['days_since_invoice']),
        "payment_term_days": int(row['payment_term_days']),
        "risk_rating": "",
        "reasoning": "",
        "overdue_flag": False,
        "days_overdue": 0,
        "amount_tier": ""
    }

    # Run it
    result = app.invoke(invoice)
    print(f"{result['vendor']:<12} | {result['amount_tier']:<12} | Overdue: {result['days_overdue']} days | Risk: {result['risk_rating']}")
    print(f"Vendor: {result['vendor']}")
    print(f"Overdue flag: {result['overdue_flag']}")
    print(f"Amount Tier: {result['amount_tier']}")
    print(f"Days Overdue: {result['days_overdue']}")
    print(f"Risk Rating: {result['risk_rating']}")
    print(f"Reasoning: {result['reasoning'][:200]}")

# After loop - show full telemetry
conn = sqlite3.connect("collections.db")
summary = pd.read_sql("""
    SELECT vendor, risk_rating, amount_tier, days_overdue, timestamp
    FROM agent_runs
    ORDER BY run_id DESC
    LIMIT 10
""", conn)
conn.close()
print("\nTelemetry Summary:")
print(summary)