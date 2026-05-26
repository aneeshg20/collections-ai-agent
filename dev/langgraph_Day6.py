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
    aging_risk_flag: bool
    aging_summary: str
    dispute_count: int
    dispute_flag: bool

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

def collection_strategy(state: InvoiceState):
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
    Aging Risk Flag: {state['aging_risk_flag']}
    Aging Summary: {state['aging_summary']}
    Active Disputes: {state['dispute_count']}
    Dispute Flag: {state['dispute_flag']}

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

def check_ageing(state: InvoiceState):
    df = pd.read_csv("agent/aging_buckets.csv")
    vendor_data = df[df['vendor'] == state['vendor']]
    if vendor_data.empty:
        return{"aging_risk_flag": False, "aging_sumary": "No aging data found"}
    bucket_61_90 = vendor_data['bucket_61_90'].iloc[0]
    bucket_90_plus = vendor_data['bucket_90_plus'].iloc[0]
    bucket_31_60 = vendor_data['bucket_31_60'].iloc[0]

    aging_risk_flag = bucket_61_90 > 0 or bucket_90_plus > 0

    aging_summary = f"0-30: {vendor_data['bucket_0_30'].iloc[0]:,} | 31-60: {bucket_31_60:,} | 61-90: {bucket_61_90:,} | 90+: {bucket_90_plus:,}"
    return {
                "aging_risk_flag": aging_risk_flag,
                "aging_summary": aging_summary
    }

def check_disputes(state: InvoiceState):
    df = pd.read_csv("agent/dispute_history.csv")
    vendor_disputes = df[df['vendor'] == state['vendor']]
    open_disputes = vendor_disputes[vendor_disputes['status'] == "Open"]
    count_open_disputes = len(open_disputes)
    dispute_flag = count_open_disputes > 0
    return {
                "dispute_flag": dispute_flag,
                "dispute_count": count_open_disputes
    }


def route_by_risk(state:InvoiceState) ->str:
    if state['risk_rating'] == "HIGH":
        return "escalate"
    else:
        return "log_telemetry"

# Build the graph
graph = StateGraph(InvoiceState)
graph.add_node("check_ageing", check_ageing)
graph.add_node("check_disputes", check_disputes)
graph.add_node("pre_classify", pre_classify)
graph.add_edge(START, "check_ageing")
graph.add_edge("check_ageing","check_disputes")
graph.add_edge("check_disputes","pre_classify")
graph.add_node("collection_strategy", collection_strategy)
graph.add_edge("pre_classify","collection_strategy")
graph.add_node("escalate", escalate)
graph.add_node("log_telemetry", log_telemetry)
graph.add_conditional_edges(
    "collection_strategy",        # from this node
    route_by_risk,        # use this function to decide
    {
        "escalate": "escalate",           # if returns "escalate"
        "log_telemetry": "log_telemetry"  # if returns "log_telemetry"
    }
)
graph.add_edge("escalate","log_telemetry")
graph.add_edge("log_telemetry", END)
app = graph.compile()
# Visualise the graph
from IPython.display import Image
graph_image = app.get_graph().draw_mermaid_png()
with open("agent/graph_visualisation.png", "wb") as f:
    f.write(graph_image)
print("Graph saved to agent/graph_visualisation.png")

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
        "amount_tier": "",
        "aging_risk_flag": False,   
        "aging_summary": "",         
        "dispute_count": 0,          
        "dispute_flag": False        
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