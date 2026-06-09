import streamlit as st
import pandas as pd
import sqlite3
import json
import numpy as np
import hashlib
import os
import time
from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
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
    drafted_communication: str

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
# Utility Agent Nodes (with defensive handling)
# ============================================================

def check_ageing(state: InvoiceState):
    try:
        df = pd.read_csv("agent/aging_buckets.csv")
        vendor_data = df[df['vendor'] == state['vendor']]
        if vendor_data.empty:
            return {"aging_risk_flag": False, "aging_summary": f"No aging data for {state['vendor']}"}
        bucket_61_90 = vendor_data['bucket_61_90'].iloc[0]
        bucket_90_plus = vendor_data['bucket_90_plus'].iloc[0]
        bucket_31_60 = vendor_data['bucket_31_60'].iloc[0]
        aging_risk_flag = bucket_61_90 > 0 or bucket_90_plus > 0
        aging_summary = f"0-30: {vendor_data['bucket_0_30'].iloc[0]:,} | 31-60: {bucket_31_60:,} | 61-90: {bucket_61_90:,} | 90+: {bucket_90_plus:,}"
        return {"aging_risk_flag": aging_risk_flag, "aging_summary": aging_summary}
    except Exception as e:
        return {"aging_risk_flag": False, "aging_summary": f"Error reading aging data: {str(e)[:50]}"}

def check_disputes(state: InvoiceState):
    try:
        df = pd.read_csv("agent/dispute_history.csv")
        vendor_disputes = df[df['vendor'] == state['vendor']]
        open_disputes = vendor_disputes[vendor_disputes['status'] == "Open"]
        count_open_disputes = len(open_disputes)
        dispute_flag = count_open_disputes > 0
        return {"dispute_flag": dispute_flag, "dispute_count": count_open_disputes}
    except Exception as e:
        return {"dispute_flag": False, "dispute_count": 0}

def check_ptp(state: InvoiceState):
    try:
        df = pd.read_csv("agent/ptp_history.csv")
        vendor_ptp = df[df['vendor'] == state['vendor']]
        ptp_broken_frame = vendor_ptp[vendor_ptp['broken_flag'] == True]
        count_ptp_broken = len(ptp_broken_frame)
        ptp_broken_flag = count_ptp_broken >= 2
        return {"ptp_broken_count": count_ptp_broken, "ptp_risk_flag": ptp_broken_flag}
    except Exception as e:
        return {"ptp_broken_count": 0, "ptp_risk_flag": False}

def check_credit(state: InvoiceState):
    try:
        df = pd.read_csv("agent/credit_balance.csv")
        vendor_utilization = df[df['vendor'] == state['vendor']]
        if vendor_utilization.empty:
            return {"credit_utilization": 0.0, "credit_risk_flag": False}
        utilization_pct = vendor_utilization['utilisation_pct'].iloc[0]
        credit_risk_flag = utilization_pct > 80
        return {"credit_utilization": float(utilization_pct), "credit_risk_flag": credit_risk_flag}
    except Exception as e:
        return {"credit_utilization": 0.0, "credit_risk_flag": False}

def check_vendor_master(state: InvoiceState):
    try:
        df = pd.read_csv("agent/vendor_master.csv")
        vendor_master = df[df['vendor'] == state["vendor"]]
        if vendor_master.empty:
            return {"vendor_tier": "Unknown", "payment_score": 50, "avg_days_to_pay": 30, "vendor_risk_flag": False}
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
    except Exception as e:
        return {"vendor_tier": "Unknown", "payment_score": 50, "avg_days_to_pay": 30, "vendor_risk_flag": False}

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
    except Exception as e:
        return {"retrieved_context": "No historical data available"}

def collection_strategy(state: InvoiceState):
    try:
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
    except Exception as e:
        return {
            "risk_rating": "UNKNOWN",
            "reasoning": f"Error in strategy agent: {str(e)[:100]}",
            "recommended_action": "SEND_COURTESY_REMINDER"
        }

def draft_communication(state: InvoiceState):
    try:
        action = state["recommended_action"]
        
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
        else:
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
Amount: INR {state['invoice_amount']:,.0f}
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
            messages=[{"role": "user", "content": f"Draft the {comm_type} for {state['vendor']}."}]
        )
        
        return {"drafted_communication": message.content[0].text}
    except Exception as e:
        return {"drafted_communication": f"Error drafting communication: {str(e)[:100]}"}

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

def orchestrator_dispatch(state: InvoiceState):
    """Dispatches all 5 UAs in parallel via Send API."""
    return [
        Send("check_ageing", state),
        Send("check_disputes", state),
        Send("check_ptp", state),
        Send("check_credit", state),
        Send("check_vendor_master", state)
    ]

def aggregator(state: InvoiceState):
    """Pass-through node - waits for all parallel UAs to complete."""
    return {}

def route_by_risk(state: InvoiceState) -> str:
    if state['risk_rating'] == "HIGH":
        return "escalate"
    else:
        return "log_telemetry"

# ============================================================
# Build the Graph (with Orchestrator pattern)
# ============================================================

@st.cache_resource
def build_graph():
    graph = StateGraph(InvoiceState)
    graph.add_node("check_ageing", check_ageing)
    graph.add_node("check_disputes", check_disputes)
    graph.add_node("check_ptp", check_ptp)
    graph.add_node("check_credit", check_credit)
    graph.add_node("check_vendor_master", check_vendor_master)
    graph.add_node("aggregator", aggregator)
    graph.add_node("pre_classify", pre_classify)
    graph.add_node("retrieve_history", retrieve_history)
    graph.add_node("collection_strategy", collection_strategy)
    graph.add_node("draft_communication", draft_communication)
    graph.add_node("escalate", escalate)
    graph.add_node("log_telemetry", log_telemetry)
    
    # Orchestrator: START dispatches all 5 UAs in parallel
    graph.add_conditional_edges(
        START,
        orchestrator_dispatch,
        ["check_ageing", "check_disputes", "check_ptp", 
         "check_credit", "check_vendor_master"]
    )
    
    # All 5 UAs flow into aggregator (parallel merge point)
    graph.add_edge("check_ageing", "aggregator")
    graph.add_edge("check_disputes", "aggregator")
    graph.add_edge("check_ptp", "aggregator")
    graph.add_edge("check_credit", "aggregator")
    graph.add_edge("check_vendor_master", "aggregator")
    
    graph.add_edge("aggregator", "pre_classify")
    graph.add_edge("pre_classify", "retrieve_history")
    graph.add_edge("retrieve_history", "collection_strategy")
    graph.add_edge("collection_strategy", "draft_communication")
    
    graph.add_conditional_edges(
        "draft_communication",
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
# CSV Validation
# ============================================================

REQUIRED_COLUMNS = ['vendor', 'invoice_amount', 'days_since_invoice', 'payment_term_days']
MAX_INVOICES = 50  # Safety limit to prevent API cost overruns

def validate_csv(df):
    """Returns (is_valid, error_message)"""
    if df.empty:
        return False, "❌ The uploaded CSV is empty. Please upload a file with invoice data."
    
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        return False, f"❌ Missing required columns: {missing_cols}. Required: {REQUIRED_COLUMNS}"
    
    if len(df) > MAX_INVOICES:
        return False, f"❌ Too many invoices ({len(df)}). Maximum allowed: {MAX_INVOICES} (to control API costs in this demo)."
    
    # Check for nulls in required columns
    for col in REQUIRED_COLUMNS:
        if df[col].isnull().any():
            return False, f"❌ Column '{col}' contains empty values. Please clean your data."
    
    # Check numeric columns are actually numeric
    for col in ['invoice_amount', 'days_since_invoice', 'payment_term_days']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"❌ Column '{col}' must contain numbers only."
    
    return True, ""

# ============================================================
# Streamlit UI
# ============================================================

st.title("💰 Collections AI Agent")
st.markdown("**Enterprise Invoice Risk Assessment | Powered by 12-node LangGraph pipeline with Orchestrator + RAG**")

st.warning("⚠️ This is a portfolio demo. Each assessment uses Claude API credits. Limited to 50 invoices per upload.")

with st.sidebar:
    st.header("📋 Architecture")
    st.markdown("""
    **Pipeline (12 nodes):**
    
    *Parallel UA Layer (Orchestrator):*
    1. Check Ageing UA
    2. Check Disputes UA
    3. Check PTP History UA
    4. Check Credit Balance UA
    5. Check Vendor Master UA
    
    *Sequential Strategy Layer:*
    6. Aggregator
    7. Pre-Classifier
    8. Retrieve History (RAG)
    9. Collection Strategy Agent
    10. Communication Drafting Agent
    11. Conditional Escalation
    12. Telemetry Logger
    """)
    
    st.header("🛠️ Tech Stack")
    st.markdown("""
    - Claude Sonnet 4.5 API
    - LangGraph (Send API)
    - SQLite (telemetry)
    - Vector Store (RAG)
    - Pandas
    - Streamlit
    """)

st.subheader("📤 Upload Invoice Data")
uploaded_file = st.file_uploader("Upload invoice CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        invoices_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Could not read CSV file: {str(e)}")
        st.stop()
    
    # Validate CSV
    is_valid, error_msg = validate_csv(invoices_df)
    if not is_valid:
        st.error(error_msg)
        st.info(f"**Required columns:** {REQUIRED_COLUMNS}")
        st.stop()
    
    st.subheader("📋 Uploaded Invoices Preview")
    st.dataframe(invoices_df, use_container_width=True)
    st.caption(f"Total invoices: {len(invoices_df)}")
    
    if st.button("🚀 Run Risk Assessment", type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        errors = []
        
        for index, row in invoices_df.iterrows():
            status_text.text(f"Processing {row['vendor']}... ({index + 1}/{len(invoices_df)})")
            
            try:
                invoice = {
                    "vendor": str(row['vendor']),
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
                    "retrieved_context": "",
                    "drafted_communication": ""
                }
                
                result = app_graph.invoke(invoice)
                results.append(result)
            except Exception as e:
                errors.append(f"Row {index+1} ({row['vendor']}): {str(e)[:100]}")
                continue
            
            progress_bar.progress((index + 1) / len(invoices_df))
        
        status_text.text(f"✅ Processed {len(results)} invoices successfully")
        
        if errors:
            with st.expander(f"⚠️ {len(errors)} errors occurred"):
                for err in errors:
                    st.text(err)
        
        if not results:
            st.error("No invoices were processed successfully. Please check your data.")
            st.stop()
        
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
        
        st.subheader("📊 Detailed Risk Assessment")
        st.caption("Click any vendor to see full analysis and drafted communication")
        
        # ✅ FIX: Use idx in keys to handle duplicate vendors
        for idx, result in enumerate(results):
            color = "🔴" if result['risk_rating'] == 'HIGH' else "🟡" if result['risk_rating'] == 'MEDIUM' else "🟢"
            
            with st.expander(
                f"{color} **Row {idx+1}: {result['vendor']}** | Risk: **{result['risk_rating']}** | Action: **{result['recommended_action']}**"
            ):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Amount", f"₹{result['invoice_amount']:,.0f}")
                col2.metric("Days Overdue", result['days_overdue'])
                col3.metric("Vendor Tier", result['vendor_tier'])
                col4.metric("Payment Score", f"{result['payment_score']}/100")
                
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
                
                # ✅ FIX: All keys now include idx for uniqueness
                tab1, tab2, tab3 = st.tabs(["💭 Full Assessment", "✉️ Drafted Communication", "📚 Historical Context"])
                
                with tab1:
                    st.markdown("**Collection Strategy Agent Output:**")
                    st.text_area("", value=result['reasoning'], height=300, key=f"reasoning_{idx}_{result['vendor']}", label_visibility="collapsed")
                
                with tab2:
                    st.markdown("**Communication Drafting Agent Output (Awaiting Human Approval):**")
                    st.markdown(result['drafted_communication'])
                    
                    col_x, col_y, col_z = st.columns(3)
                    with col_x:
                        st.button("✅ Approve & Send", key=f"approve_{idx}_{result['vendor']}", type="primary")
                    with col_y:
                        st.button("✏️ Edit Draft", key=f"edit_{idx}_{result['vendor']}")
                    with col_z:
                        st.button("❌ Reject", key=f"reject_{idx}_{result['vendor']}")
                    st.caption("Note: Approval buttons are placeholders for HITL integration (Phase v1.5)")
                
                with tab3:
                    st.markdown("**Retrieved Similar Past Cases (RAG):**")
                    st.text_area("", value=result['retrieved_context'], height=200, key=f"context_{idx}_{result['vendor']}", label_visibility="collapsed")

else:
    st.info("👆 Upload an invoices CSV file to start the assessment.")
    
    st.subheader("📋 Required CSV Format")
    st.markdown(f"**Required columns:** `{', '.join(REQUIRED_COLUMNS)}`")
    sample = pd.DataFrame({
        "vendor": ["Honeywell", "Siemens", "ABB"],
        "invoice_amount": [1200000, 500000, 750000],
        "days_since_invoice": [67, 25, 45],
        "payment_term_days": [45, 30, 60]
    })
    st.dataframe(sample, use_container_width=True)
    st.caption(f"Maximum {MAX_INVOICES} invoices per upload. Vendors should match supporting data files in the repo.")