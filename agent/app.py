import streamlit as st
import pandas as pd
import sqlite3
import json
import numpy as np
import hashlib
import os
from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from datetime import datetime

# ============================================================
# Setup
# ============================================================

load_dotenv()
client = Anthropic()

st.set_page_config(
    page_title="Collections AI Agent",
    page_icon="💰",
    layout="wide"
)

# ============================================================
# State Definition
# ============================================================

class InvoiceState(TypedDict):
    vendor: str
    invoice_amount: float
    days_since_invoice: int
    payment_term_days: int
    risk_rating: str
    reasoning: str
    overdue_flag: bool
    days_overdue: int
    amount_tier: str
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

# ============================================================
# Helper Functions
# ============================================================

def embed(text):
    hash_bytes = hashlib.sha256(text.encode()).digest()
    return [hash_bytes[i % len(hash_bytes)] / 255.0 for i in range(128)]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ============================================================
# Utility Agent Nodes
# ============================================================

def check_ageing(state: InvoiceState):
    df = pd.read_csv("agent/aging_buckets.csv")
    vendor_data = df[df['vendor'] == state['vendor']]
    if vendor_data.empty:
        return {"aging_risk_flag": False, "aging_summary": "No aging data"}
    bucket_61_90 = vendor_data['bucket_61_90'].iloc[0]
    bucket_90_plus = vendor_data['bucket_90_plus'].iloc[0]
    bucket_31_60 = vendor_data['bucket_31_60'].iloc[0]
    aging_risk_flag = bucket_61_90 > 0 or bucket_90_plus > 0
    aging_summary = f"0-30: {vendor_data['bucket_0_30'].iloc[0]:,} | 31-60: {bucket_31_60:,} | 61-90: {bucket_61_90:,} | 90+: {bucket_90_plus:,}"
    return {"aging_risk_flag": aging_risk_flag, "aging_summary": aging_summary}

def check_disputes(state: InvoiceState):
    df = pd.read_csv("agent/dispute_history.csv")
    vendor_disputes = df[df['vendor'] == state['vendor']]
    open_disputes = vendor_disputes[vendor_disputes['status'] == "Open"]
    count_open_disputes = len(open_disputes)
    dispute_flag = count_open_disputes > 0
    return {"dispute_flag": dispute_flag, "dispute_count": count_open_disputes}

def check_ptp(state: InvoiceState):
    df = pd.read_csv("agent/ptp_history.csv")
    vendor_ptp = df[df['vendor'] == state['vendor']]
    ptp_broken_frame = vendor_ptp[vendor_ptp['broken_flag'] == True]
    count_ptp_broken = len(ptp_broken_frame)
    ptp_broken_flag = count_ptp_broken >= 2
    return {"ptp_broken_count": count_ptp_broken, "ptp_risk_flag": ptp_broken_flag}

def check_credit(state: InvoiceState):
    df = pd.read_csv("agent/credit_balance.csv")
    vendor_utilization = df[df['vendor'] == state['vendor']]
    utilization_pct = vendor_utilization['utilisation_pct'].iloc[0]
    credit_risk_flag = utilization_pct > 80
    return {"credit_utilization": utilization_pct, "credit_risk_flag": credit_risk_flag}

def check_vendor_master(state: InvoiceState):
    df = pd.read_csv("agent/vendor_master.csv")
    vendor_master = df[df['vendor'] == state["vendor"]]
    vendor_tier = vendor_master['tier'].iloc[0]
    payment_score = vendor_master['payment_score'].iloc[0]
    avg_days_to_pay = vendor_master['avg_days_to_pay'].iloc[0]
    vendor_risk_flag = vendor_tier == "At-Risk" or payment_score < 50
    return {
        "vendor_tier": vendor_tier,
        "payment_score": int(payment_score),
        "avg_days_to_pay": int(avg_days_to_pay),
        "vendor_risk_flag": vendor_risk_flag
    }

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

def retrieve_history(state: InvoiceState):
    try:
        with open("vector_store.json", "r") as f:
            vector_store = json.load(f)
    except FileNotFoundError:
        return {"retrieved_context": "No historical data available"}
    
    query_text = f"Vendor {state['vendor']} amount {state['invoice_amount']} days overdue {state['days_overdue']} risk assessment"
    query_embedding = embed(query_text)
    
    scored = []
    for item in vector_store:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append((score, item))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    top_3 = scored[:3]
    
    context_parts = []
    for score, item in top_3:
        context_parts.append(f"[Similarity: {score:.2f}] {item['document'][:200]}")
    retrieved_context = "\n\n".join(context_parts)
    
    return {"retrieved_context": retrieved_context}

def collection_strategy(state: InvoiceState):
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=f"""You are the Collections Strategy Agent for a world class managed services collections team.

Score each signal using the rubric below, total the points, determine the risk rating, and recommend a precise action.

SCORING RUBRIC:
- Aging Risk Flag = True         -> +2 points
- Disputes 2 or more open        -> +3 points
- Disputes 1 open                -> +2 points
- PTP Broken 3 or more           -> +3 points
- PTP Broken 1 to 2              -> +2 points
- Credit Utilisation above 90%   -> +3 points
- Credit Utilisation 70 to 90%   -> +2 points
- Vendor Risk Flag = True        -> +2 points
- Days Overdue above 30          -> +3 points
- Days Overdue 1 to 30           -> +1 point

THRESHOLDS:
0 to 3 points  -> LOW
4 to 7 points  -> MEDIUM
8 or more      -> HIGH

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
HISTORICAL CONTEXT: {state['retrieved_context']}

OUTPUT FORMAT - follow exactly:
RISK SCORE: [X]/15
RISK RATING: [HIGH or MEDIUM or LOW]
RECOMMENDED ACTION: [one action from the list above]
COMMUNICATION TONE: [Urgent or Firm or Friendly]
NEXT REVIEW: [X days]
REASONING: [2-3 sentences explaining key drivers]
""",
        messages=[{"role": "user", "content": f"Assess collection risk for: {state}"}]
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

    recommended_action = "SEND_COURTESY_REMINDER"
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

def escalate(state: InvoiceState):
    return {}

def log_telemetry(state: InvoiceState):
    try:
        conn = sqlite3.connect("collections.db")
        conn.execute("""
            INSERT INTO agent_runs(vendor, risk_rating, timestamp, tokens_used, model, days_overdue, amount_tier, overdue_flag, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (state['vendor'], state['risk_rating'], 
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             512, "claude-sonnet-4-5",
             state['days_overdue'], state['amount_tier'],
             state['overdue_flag'], state['reasoning']))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Telemetry logging error: {e}")
    return {}

def route_by_risk(state: InvoiceState) -> str:
    if state['risk_rating'] == "HIGH":
        return "escalate"
    else:
        return "log_telemetry"

# ============================================================
# Build the Graph
# ============================================================

@st.cache_resource
def build_graph():
    graph = StateGraph(InvoiceState)
    graph.add_node("check_ageing", check_ageing)
    graph.add_node("check_disputes", check_disputes)
    graph.add_node("check_ptp", check_ptp)
    graph.add_node("check_credit", check_credit)
    graph.add_node("check_vendor_master", check_vendor_master)
    graph.add_node("pre_classify", pre_classify)
    graph.add_node("retrieve_history", retrieve_history)
    graph.add_node("collection_strategy", collection_strategy)
    graph.add_node("escalate", escalate)
    graph.add_node("log_telemetry", log_telemetry)
    
    graph.add_edge(START, "check_ageing")
    graph.add_edge("check_ageing", "check_disputes")
    graph.add_edge("check_disputes", "check_ptp")
    graph.add_edge("check_ptp", "check_credit")
    graph.add_edge("check_credit", "check_vendor_master")
    graph.add_edge("check_vendor_master", "pre_classify")
    graph.add_edge("pre_classify", "retrieve_history")
    graph.add_edge("retrieve_history", "collection_strategy")
    
    graph.add_conditional_edges(
        "collection_strategy",
        route_by_risk,
        {
            "escalate": "escalate",
            "log_telemetry": "log_telemetry"
        }
    )
    
    graph.add_edge("escalate", "log_telemetry")
    graph.add_edge("log_telemetry", END)
    
    return graph.compile()

app_graph = build_graph()

# ============================================================
# Streamlit UI
# ============================================================

st.title("💰 Collections AI Agent")
st.markdown("**Enterprise Invoice Risk Assessment | Powered by 10-node LangGraph pipeline with RAG**")

# Sidebar info
with st.sidebar:
    st.header("📋 Architecture")
    st.markdown("""
    **Pipeline:**
    1. Check Ageing UA
    2. Check Disputes UA
    3. Check PTP History UA
    4. Check Credit Balance UA
    5. Check Vendor Master UA
    6. Pre-Classifier
    7. Retrieve History (RAG)
    8. Collection Strategy Agent
    9. Conditional Escalation
    10. Telemetry Logger
    """)
    
    st.header("🛠️ Tech Stack")
    st.markdown("""
    - Claude Sonnet 4.5 API
    - LangGraph
    - SQLite (telemetry)
    - ChromaDB (RAG)
    - Pandas
    - Streamlit
    """)

# Main content
st.subheader("📤 Upload Invoice Data")
uploaded_file = st.file_uploader("Upload invoice CSV file", type=['csv'])

if uploaded_file is not None:
    invoices_df = pd.read_csv(uploaded_file)
    
    st.subheader("📋 Uploaded Invoices Preview")
    st.dataframe(invoices_df, use_container_width=True)
    
    if st.button("🚀 Run Risk Assessment", type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for index, row in invoices_df.iterrows():
            status_text.text(f"Processing {row['vendor']}... ({index + 1}/{len(invoices_df)})")
            
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
                "credit_utilization": 0.0,
                "credit_risk_flag": False,
                "vendor_tier": "",
                "payment_score": 0,
                "vendor_risk_flag": False,
                "avg_days_to_pay": 0,
                "recommended_action": "",
                "retrieved_context": ""
            }
            
            result = app_graph.invoke(invoice)
            results.append(result)
            
            progress_bar.progress((index + 1) / len(invoices_df))
        
        status_text.text("✅ Processing complete!")
        
        # Portfolio Summary Metrics
        st.subheader("📊 Portfolio Summary")
        
        high_risk = sum(1 for r in results if r['risk_rating'] == 'HIGH')
        medium_risk = sum(1 for r in results if r['risk_rating'] == 'MEDIUM')
        low_risk = sum(1 for r in results if r['risk_rating'] == 'LOW')
        total_amount = sum(r['invoice_amount'] for r in results)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Invoices", len(results))
        col2.metric("Total Exposure", f"₹{total_amount:,.0f}")
        col3.metric("🔴 HIGH Risk", high_risk)
        col4.metric("🟡 MEDIUM Risk", medium_risk)
        
        # Detailed Results
        st.subheader("📊 Detailed Risk Assessment")
        st.caption("Click any vendor to see full analysis")
        
        for result in results:
            color = "🔴" if result['risk_rating'] == 'HIGH' else "🟡" if result['risk_rating'] == 'MEDIUM' else "🟢"
            
            with st.expander(
                f"{color} **{result['vendor']}** | Risk: **{result['risk_rating']}** | Action: **{result['recommended_action']}**"
            ):
                # Top metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Amount", f"₹{result['invoice_amount']:,.0f}")
                col2.metric("Days Overdue", result['days_overdue'])
                col3.metric("Vendor Tier", result['vendor_tier'])
                col4.metric("Payment Score", f"{result['payment_score']}/100")
                
                # UA Findings
                st.markdown("### 📋 Utility Agent Findings")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Aging Risk:** {'🔴 Flagged' if result['aging_risk_flag'] else '🟢 Clean'}")
                    st.caption(result['aging_summary'])
                    st.markdown(f"**Open Disputes:** {result['dispute_count']} {'🔴' if result['dispute_flag'] else '🟢'}")
                    st.markdown(f"**PTP Broken:** {result['ptp_broken_count']} {'🔴' if result['ptp_risk_flag'] else '🟢'}")
                
                with col_b:
                    st.markdown(f"**Credit Utilization:** {result['credit_utilization']}% {'🔴' if result['credit_risk_flag'] else '🟢'}")
                    st.markdown(f"**Vendor Risk Flag:** {'🔴 At-Risk' if result['vendor_risk_flag'] else '🟢 Healthy'}")
                    st.markdown(f"**Avg Days to Pay:** {result['avg_days_to_pay']}")
                
                # Full Reasoning
                st.markdown("### 💭 Full Risk Assessment")
                st.text_area("", value=result['reasoning'], height=300, key=f"reasoning_{result['vendor']}", label_visibility="collapsed")
                
                # RAG Context
                st.markdown("### 📚 Historical Context (Retrieved via RAG)")
                st.text_area("", value=result['retrieved_context'], height=150, key=f"context_{result['vendor']}", label_visibility="collapsed")

else:
    st.info("👆 Upload an invoices CSV file to start the assessment. Use the format: vendor, invoice_amount, days_since_invoice, payment_term_days")
    
    # Show sample data
    st.subheader("📋 Sample Format")
    sample = pd.DataFrame({
        "vendor": ["Honeywell", "Siemens"],
        "invoice_amount": [1200000, 500000],
        "days_since_invoice": [67, 25],
        "payment_term_days": [45, 30]
    })
    st.dataframe(sample, use_container_width=True)