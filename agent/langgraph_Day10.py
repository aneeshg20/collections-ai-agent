from anthropic import Anthropic
from dotenv import load_dotenv
from datetime import datetime
import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import sqlite3
import pandas as pd
import json
import numpy as np
import hashlib

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
    ptp_broken_count: int
    ptp_risk_flag: bool
    credit_utilization: float
    credit_risk_flag: bool
    vendor_tier: str
    payment_score: int
    vendor_risk_flag: bool
    avg_days_to_pay: int
    recommended_action: str
    retrieved_context: str
    drafted_communication: str


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
    system = f"""You are the Collections Strategy Agent for a world class managed services collections team.

    Score each signal using the rubric below, total the points, determine the risk rating, and recommend a precise action.

    SCORING RUBRIC:
    - Aging Risk Flag = True          → +2 points
    - Disputes 2 or more open        → +3 points
    - Disputes 1 open                → +2 points
    - PTP Broken 3 or more           → +3 points
    - PTP Broken 1 to 2              → +2 points
    - Credit Utilisation above 90%   → +3 points
    - Credit Utilisation 70 to 90%   → +2 points
    - Vendor Risk Flag = True        → +2 points
    - Days Overdue above 30          → +3 points
    - Days Overdue 1 to 30           → +1 point

    THRESHOLDS:
    0 to 3 points  → LOW
    4 to 7 points  → MEDIUM
    8 or more      → HIGH

    RECOMMENDED ACTIONS (pick exactly one):
    - SEND_COURTESY_REMINDER
    - SEND_DUNNING_LEVEL_1
    - SEND_DUNNING_LEVEL_2
    - ESCALATE_SENIOR_MANAGEMENT
    - PLACE_CREDIT_HOLD
    - INITIATE_LEGAL_REVIEW

    INPUT SIGNALS:
    Vendor: {state['vendor']}
    Amount: {state['invoice_amount']} ({state['amount_tier']})
    Days Overdue: {state['days_overdue']}
    Aging Risk: {state['aging_risk_flag']} | {state['aging_summary']}
    Open Disputes: {state['dispute_count']} | Flag: {state['dispute_flag']}
    PTP Broken Count: {state['ptp_broken_count']} | Flag: {state['ptp_risk_flag']}
    Credit Utilization: {state['credit_utilization']}% | Flag: {state['credit_risk_flag']}
    Vendor Tier: {state['vendor_tier']} | Payment Score: {state['payment_score']}/100 | Flag: {state['vendor_risk_flag']}
    Avg Days to Pay: {state['avg_days_to_pay']}
    HISTORICAL CONTEXT (similar past cases): {state['retrieved_context']}

    OUTPUT FORMAT — follow exactly:
    RISK SCORE: [X]/15
    RISK RATING: [HIGH or MEDIUM or LOW]
    RECOMMENDED ACTION: [one action from the list above]
    COMMUNICATION TONE: [Urgent or Firm or Friendly]
    NEXT REVIEW: [X days]
    REASONING: [2-3 sentences explaining key drivers]
    """,
    messages=[
        {"role":"user", "content": f" Assess collection risk for this invoice: {state}"}
    ]
    )
    response_text = message.content[0].text

    if "HIGH" in response_text:
        rating = "HIGH"
    elif "MEDIUM" in response_text:
        rating = "MEDIUM"
    elif "LOW" in response_text:
        rating = "LOW"
    else:
        rating = "UNKNOWN"

    # Extract recommended action
    recommended_action = "SEND_COURTESY_REMINDER"  # default
    for action in ["ESCALATE_SENIOR_MANAGEMENT", "PLACE_CREDIT_HOLD", 
                    "INITIATE_LEGAL_REVIEW", "SEND_DUNNING_LEVEL_2",
                    "SEND_DUNNING_LEVEL_1", "SEND_COURTESY_REMINDER"]:
        if action in response_text:
            recommended_action = action
            break

    return {
        "risk_rating": rating,
        "reasoning": response_text,
        "recommended_action": recommended_action
    }


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

def check_ptp(state: InvoiceState):
    df = pd.read_csv("agent/ptp_history.csv")
    vendor_ptp = df[df['vendor'] == state['vendor']]
    ptp_broken_frame = vendor_ptp[vendor_ptp['broken_flag'] == True]
    count_ptp_broken = len(ptp_broken_frame)
    ptp_broken_flag = count_ptp_broken >= 2
    return{
            "ptp_broken_count": count_ptp_broken,
            "ptp_risk_flag": ptp_broken_flag
    }

def check_credit(state: InvoiceState):
    df = pd.read_csv("agent/credit_balance.csv")
    vendor_utilization = df[df['vendor'] == state['vendor']]
    utilization_pct = vendor_utilization['utilisation_pct'].iloc[0]
    credit_risk_flag = utilization_pct > 80
    return{
            "credit_utilization": utilization_pct,
            "credit_risk_flag": credit_risk_flag
    }

def check_vendor_master(state: InvoiceState):
    df = pd.read_csv("agent/vendor_master.csv")
    vendor_master = df[df['vendor'] == state["vendor"]]
    vendor_tier = vendor_master['tier'].iloc[0]
    payment_score = vendor_master['payment_score'].iloc[0]
    avg_days_to_pay = vendor_master['avg_days_to_pay'].iloc[0]
    vendor_risk_flag = vendor_tier == "At-Risk" or payment_score < 50
    return{
            "vendor_tier": vendor_tier,
            "payment_score": payment_score,
            "avg_days_to_pay": avg_days_to_pay,
            "vendor_risk_flag": vendor_risk_flag
    }

def embed(text):
    """Same hash embedding as chromadb_setup.py"""
    hash_bytes = hashlib.sha256(text.encode()).digest()
    return [hash_bytes[i % len(hash_bytes)] / 255.0 for i in range(128)]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def retrieve_history(state: InvoiceState):
    # Load vector store
    with open("vector_store.json", "r") as f:
        vector_store = json.load(f)
    
    # Build query text from current invoice context
    query_text = f"Vendor {state['vendor']} amount {state['invoice_amount']} days overdue {state['days_overdue']} risk assessment"
    
    # Embed the query
    query_embedding = embed(query_text)
    
    # Score all stored documents
    scored = []
    for item in vector_store:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append((score, item))
    
    # Sort by similarity, take top 3
    scored.sort(key=lambda x: x[0], reverse=True)
    top_3 = scored[:3]
    
    # Build context string for Claude
    context_parts = []
    for score, item in top_3:
        context_parts.append(
            f"[Similarity: {score:.2f}] {item['document'][:200]}"
        )
    retrieved_context = "\n\n".join(context_parts)
    
    return {"retrieved_context": retrieved_context}

def draft_communication (state: InvoiceState):
    #Map action to communication style
    action = state["recommended_action"]

    #Determine template type
    # Determine template type
    if action == "ESCALATE_SENIOR_MANAGEMENT":
        comm_type = "internal escalation memo to senior management"
        tone = "urgent and factual"
        recipient = "CFO and Head of Finance"
    elif action == "PLACE_CREDIT_HOLD":
        comm_type = "internal credit hold notification + customer notification"
        tone = "firm but professional"
        recipient = "Credit team and customer AP contact"
    elif action == "INITIATE_LEGAL_REVIEW":
        comm_type = "legal review request memo"
        tone = "formal and detailed"
        recipient = "Legal department"
    elif action == "SEND_DUNNING_LEVEL_2":
        comm_type = "firm dunning email - Level 2"
        tone = "firm with clear payment deadline"
        recipient = "Customer AP contact"
    elif action == "SEND_DUNNING_LEVEL_1":
        comm_type = "standard dunning email - Level 1"
        tone = "professional reminder"
        recipient = "Customer AP contact"
    else:  # SEND_COURTESY_REMINDER
        comm_type = "courtesy payment reminder"
        tone = "friendly and brief"
        recipient = "Customer AP contact"

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=f"""You are the Communication Drafting Agent for an enterprise collections team.

    Draft a {comm_type} for the following invoice situation.

    RECIPIENT: {recipient}
    TONE: {tone}

    INVOICE CONTEXT:
    Vendor: {state['vendor']}
    Amount: ₹{state['invoice_amount']:,.0f}
    Days Overdue: {state['days_overdue']}
    Vendor Tier: {state['vendor_tier']}
    Risk Rating: {state['risk_rating']}
    Recommended Action: {action}

    KEY RISK SIGNALS:
    - Aging: {state['aging_summary']}
    - Open Disputes: {state['dispute_count']}
    - Broken PTPs: {state['ptp_broken_count']}
    - Credit Utilization: {state['credit_utilization']}%

    REQUIREMENTS:
    1. Include appropriate subject line
    2. Reference invoice details specifically
    3. State clear next steps with deadlines
    4. Match the tone requested
    5. End with appropriate signature line
    6. Maximum 200 words

    Output the complete communication ready for human review and approval.
    """,
            messages=[{
                "role": "user",
                "content": f"Draft the {comm_type} for {state['vendor']}."
            }]
        )
        
    return {"drafted_communication": message.content[0].text}     

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
graph.add_node("check_ptp", check_ptp)
graph.add_node("check_credit", check_credit)
graph.add_node("check_vendor_master", check_vendor_master)
graph.add_edge("check_disputes","check_ptp")
graph.add_edge("check_ptp","check_credit")
graph.add_edge("check_credit","check_vendor_master",)
graph.add_edge("check_vendor_master","pre_classify")
graph.add_node("collection_strategy", collection_strategy)
graph.add_node("retrieve_history", retrieve_history)
graph.add_edge("pre_classify","retrieve_history")
graph.add_edge("retrieve_history","collection_strategy")
graph.add_node("escalate", escalate)
graph.add_node("log_telemetry", log_telemetry)
graph.add_node("draft_communication", draft_communication)
graph.add_edge("collection_strategy", "draft_communication")
graph.add_conditional_edges(
    "draft_communication",        # from this node
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
        "dispute_flag": False,
        "ptp_broken_count": 0,
        "ptp_risk_flag": False,
        "credit_utilization": 0,
        "credit_risk_flag": False,
        "vendor_tier": "",
        "payment_score": 0,
        "vendor_risk_flag": False,
        "avg_days_to_pay": 0,
        "recommended_action": "",
        "retrieved_context": "",
        "drafted_communication": ""
    }

    # Run it
    result = app.invoke(invoice)
    print(f"{result['vendor']:<12} | {result['amount_tier']:<12} | Overdue: {result['days_overdue']} days | Risk: {result['risk_rating']}")
    print(f"Vendor: {result['vendor']}")
    print(f"Overdue flag: {result['overdue_flag']}")
    print(f"Amount Tier: {result['amount_tier']}")
    print(f"Days Overdue: {result['days_overdue']}")
    print(f"Risk Rating: {result['risk_rating']}")
    print(f"Reasoning: {result['reasoning'][:300]}")
    print(f"Recommended Action: {result['recommended_action']}")
    print(f"\n--- DRAFTED COMMUNICATION ---")  
    print(result['drafted_communication'])     
    print("--- END ---\n")                       

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