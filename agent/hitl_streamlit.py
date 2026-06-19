import streamlit as st
import pandas as pd
import sqlite3
import json
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.sqlite import SqliteSaver
from sentence_transformers import SentenceTransformer
from typing import TypedDict
from datetime import datetime

# ============================================================
# Setup
# ============================================================
load_dotenv()
client = Anthropic()

st.set_page_config(page_title="HITL Collections Demo", page_icon="✅", layout="wide")

# ============================================================
# State
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
# Semantic embedding model (cached so it loads once)
# ============================================================
@st.cache_resource
def load_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

_embed_model = load_embed_model()

def embed(text):
    return _embed_model.encode(text).tolist()

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ============================================================
# Nodes (numpy-safe casts for checkpointer serialization)
# ============================================================
def check_ageing(state: InvoiceState):
    df = pd.read_csv("agent/aging_buckets.csv")
    vd = df[df['vendor'] == state['vendor']]
    if vd.empty:
        return {"aging_risk_flag": False, "aging_summary": "No aging data"}
    b6190 = vd['bucket_61_90'].iloc[0]; b90 = vd['bucket_90_plus'].iloc[0]; b3160 = vd['bucket_31_60'].iloc[0]
    return {
        "aging_risk_flag": bool(b6190 > 0 or b90 > 0),
        "aging_summary": f"0-30: {vd['bucket_0_30'].iloc[0]:,} | 31-60: {b3160:,} | 61-90: {b6190:,} | 90+: {b90:,}"
    }

def check_disputes(state: InvoiceState):
    df = pd.read_csv("agent/dispute_history.csv")
    vd = df[df['vendor'] == state['vendor']]
    open_d = vd[vd['status'] == "Open"]
    return {"dispute_flag": bool(len(open_d) > 0), "dispute_count": int(len(open_d))}

def check_ptp(state: InvoiceState):
    df = pd.read_csv("agent/ptp_history.csv")
    vd = df[df['vendor'] == state['vendor']]
    broken = vd[vd['broken_flag'] == True]
    return {"ptp_broken_count": int(len(broken)), "ptp_risk_flag": bool(len(broken) >= 2)}

def check_credit(state: InvoiceState):
    df = pd.read_csv("agent/credit_balance.csv")
    vd = df[df['vendor'] == state['vendor']]
    if vd.empty:
        return {"credit_utilization": 0.0, "credit_risk_flag": False}
    u = vd['utilisation_pct'].iloc[0]
    return {"credit_utilization": float(u), "credit_risk_flag": bool(u > 80)}

def check_vendor_master(state: InvoiceState):
    df = pd.read_csv("agent/vendor_master.csv")
    vd = df[df['vendor'] == state["vendor"]]
    if vd.empty:
        return {"vendor_tier": "Unknown", "payment_score": 50, "avg_days_to_pay": 30, "vendor_risk_flag": False}
    tier = vd['tier'].iloc[0]; score = vd['payment_score'].iloc[0]; avg = vd['avg_days_to_pay'].iloc[0]
    return {
        "vendor_tier": str(tier),
        "payment_score": int(score),
        "avg_days_to_pay": int(avg),
        "vendor_risk_flag": bool(tier == "At-Risk" or score < 50)
    }

def pre_classify(state: InvoiceState):
    overdue = bool(state['days_since_invoice'] > state['payment_term_days'])
    days_overdue = state['days_since_invoice'] - state['payment_term_days'] if overdue else 0
    if state['invoice_amount'] > 1000000:
        tier = "HIGH VALUE"
    elif state['invoice_amount'] > 300000:
        tier = "MEDIUM VALUE"
    else:
        tier = "LOW VALUE"
    return {"overdue_flag": overdue, "days_overdue": int(days_overdue), "amount_tier": tier}

def retrieve_history(state: InvoiceState):
    try:
        with open("vector_store.json", "r") as f:
            vs = json.load(f)
    except FileNotFoundError:
        return {"retrieved_context": "No historical data available"}
    q = embed(f"Vendor {state['vendor']} amount {state['invoice_amount']} days overdue {state['days_overdue']} risk assessment")
    scored = [(cosine_similarity(q, item["embedding"]), item) for item in vs]
    scored.sort(key=lambda x: x[0], reverse=True)
    parts = [f"[Similarity: {s:.2f}] {it['document'][:200]}" for s, it in scored[:3]]
    return {"retrieved_context": "\n\n".join(parts)}

def collection_strategy(state: InvoiceState):
    message = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=1024,
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

THRESHOLDS: 0-3 LOW | 4-7 MEDIUM | 8+ HIGH

RECOMMENDED ACTIONS (pick exactly one):
SEND_COURTESY_REMINDER, SEND_DUNNING_LEVEL_1, SEND_DUNNING_LEVEL_2,
ESCALATE_SENIOR_MANAGEMENT, PLACE_CREDIT_HOLD, INITIATE_LEGAL_REVIEW

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

OUTPUT FORMAT:
RISK SCORE: [X]/15
RISK RATING: [HIGH or MEDIUM or LOW]
RECOMMENDED ACTION: [one action]
COMMUNICATION TONE: [Urgent or Firm or Friendly]
NEXT REVIEW: [X days]
REASONING: [2-3 sentences]
""",
        messages=[{"role": "user", "content": f"Assess collection risk for: {state}"}]
    )
    txt = message.content[0].text
    rating = "HIGH" if "HIGH" in txt else "MEDIUM" if "MEDIUM" in txt else "LOW" if "LOW" in txt else "UNKNOWN"
    action = "SEND_COURTESY_REMINDER"
    for a in ["ESCALATE_SENIOR_MANAGEMENT", "PLACE_CREDIT_HOLD", "INITIATE_LEGAL_REVIEW",
              "SEND_DUNNING_LEVEL_2", "SEND_DUNNING_LEVEL_1", "SEND_COURTESY_REMINDER"]:
        if a in txt:
            action = a; break
    return {"risk_rating": rating, "reasoning": txt, "recommended_action": action}

def draft_communication(state: InvoiceState):
    action = state["recommended_action"]
    table = {
        "ESCALATE_SENIOR_MANAGEMENT": ("internal escalation memo to senior management", "urgent and factual", "CFO and Head of Finance"),
        "PLACE_CREDIT_HOLD": ("credit hold notification", "firm but professional", "Credit team and customer AP contact"),
        "INITIATE_LEGAL_REVIEW": ("legal review request memo", "formal and detailed", "Legal department"),
        "SEND_DUNNING_LEVEL_2": ("firm dunning email - Level 2", "firm with clear deadline", "Customer AP contact"),
        "SEND_DUNNING_LEVEL_1": ("standard dunning email - Level 1", "professional reminder", "Customer AP contact"),
    }
    comm_type, tone, recipient = table.get(action, ("courtesy payment reminder", "friendly and brief", "Customer AP contact"))
    message = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=1024,
        system=f"""You are the Communication Drafting Agent for an enterprise collections team.
Draft a {comm_type}. RECIPIENT: {recipient}. TONE: {tone}.

INVOICE: Vendor {state['vendor']}, Amount INR {state['invoice_amount']:,.0f}, Days Overdue {state['days_overdue']}, Vendor Tier {state['vendor_tier']}, Risk {state['risk_rating']}, Action {action}.
SIGNALS: Aging {state['aging_summary']}; Disputes {state['dispute_count']}; Broken PTPs {state['ptp_broken_count']}; Credit {state['credit_utilization']}%.

Requirements: subject line, reference invoice details, clear next steps with deadlines, match tone, signature line, max 200 words.
""",
        messages=[{"role": "user", "content": f"Draft the {comm_type} for {state['vendor']}."}]
    )
    return {"drafted_communication": message.content[0].text}

def escalate(state: InvoiceState):
    return {}

def log_telemetry(state: InvoiceState):
    try:
        conn = sqlite3.connect("collections.db")
        conn.execute("""INSERT INTO agent_runs(vendor, risk_rating, timestamp, tokens_used, model, days_overdue, amount_tier, overdue_flag, reasoning, recommended_action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (state['vendor'], state['risk_rating'], datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             512, "claude-sonnet-4-5", state['days_overdue'], state['amount_tier'],
             state['overdue_flag'], state['reasoning'], state['recommended_action']))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"Telemetry error: {e}")
    return {}

def send_communication(state: InvoiceState):
    return {}

def orchestrator_dispatch(state: InvoiceState):
    return [Send("check_ageing", state), Send("check_disputes", state), Send("check_ptp", state),
            Send("check_credit", state), Send("check_vendor_master", state)]

def aggregator(state: InvoiceState):
    return {}

def route_by_risk(state: InvoiceState) -> str:
    return "escalate" if state['risk_rating'] == "HIGH" else "log_telemetry"

# ============================================================
# Build graph with checkpointer + interrupt
# ============================================================
@st.cache_resource
def build_graph():
    g = StateGraph(InvoiceState)
    for name, fn in [("check_ageing", check_ageing), ("check_disputes", check_disputes),
                     ("check_ptp", check_ptp), ("check_credit", check_credit),
                     ("check_vendor_master", check_vendor_master), ("aggregator", aggregator),
                     ("pre_classify", pre_classify), ("retrieve_history", retrieve_history),
                     ("collection_strategy", collection_strategy), ("draft_communication", draft_communication),
                     ("escalate", escalate), ("log_telemetry", log_telemetry),
                     ("send_communication", send_communication)]:
        g.add_node(name, fn)

    g.add_conditional_edges(START, orchestrator_dispatch,
        ["check_ageing", "check_disputes", "check_ptp", "check_credit", "check_vendor_master"])
    for ua in ["check_ageing", "check_disputes", "check_ptp", "check_credit", "check_vendor_master"]:
        g.add_edge(ua, "aggregator")
    g.add_edge("aggregator", "pre_classify")
    g.add_edge("pre_classify", "retrieve_history")
    g.add_edge("retrieve_history", "collection_strategy")
    g.add_edge("collection_strategy", "draft_communication")
    g.add_conditional_edges("draft_communication", route_by_risk,
        {"escalate": "escalate", "log_telemetry": "log_telemetry"})
    g.add_edge("escalate", "log_telemetry")
    g.add_edge("log_telemetry", "send_communication")
    g.add_edge("send_communication", END)

    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return g.compile(checkpointer=checkpointer, interrupt_before=["send_communication"])

app_graph = build_graph()

# ============================================================
# UI - three phase HITL
# ============================================================
st.title("✅ Collections AI Agent — Human-in-the-Loop")

if "phase" not in st.session_state:
    st.session_state.phase = "input"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "result" not in st.session_state:
    st.session_state.result = None

st.caption(f"Phase: **{st.session_state.phase}**")

# ─── PHASE 1: INPUT ───
if st.session_state.phase == "input":
    st.subheader("Enter invoice details")
    vendor = st.selectbox("Vendor", ["Honeywell", "Siemens", "ABB", "Bosch", "Schneider"])
    amount = st.number_input("Invoice amount", value=1200000, step=10000)
    days_since = st.number_input("Days since invoice", value=67, step=1)
    terms = st.number_input("Payment term days", value=45, step=1)

    if st.button("🚀 Run Assessment", type="primary"):
        invoice = {
            "vendor": vendor, "invoice_amount": float(amount),
            "days_since_invoice": int(days_since), "payment_term_days": int(terms),
            "risk_rating": "", "reasoning": "", "overdue_flag": False, "days_overdue": 0,
            "amount_tier": "", "aging_risk_flag": False, "aging_summary": "",
            "dispute_count": 0, "dispute_flag": False, "ptp_broken_count": 0, "ptp_risk_flag": False,
            "credit_utilization": 0.0, "credit_risk_flag": False, "vendor_tier": "",
            "payment_score": 0, "vendor_risk_flag": False, "avg_days_to_pay": 0,
            "recommended_action": "", "retrieved_context": "", "drafted_communication": ""
        }
        thread_id = f"{vendor}_{datetime.now().strftime('%H%M%S')}"
        config = {"configurable": {"thread_id": thread_id}}

        with st.spinner("Agent processing through 12 nodes..."):
            result = app_graph.invoke(invoice, config)  # runs until interrupt_before send

        st.session_state.thread_id = thread_id
        st.session_state.result = result
        st.session_state.phase = "paused"
        st.rerun()

# ─── PHASE 2: PAUSED (awaiting approval) ───
elif st.session_state.phase == "paused":
    r = st.session_state.result
    color = "🔴" if r['risk_rating'] == 'HIGH' else "🟡" if r['risk_rating'] == 'MEDIUM' else "🟢"
    st.subheader(f"{color} {r['vendor']} — {r['risk_rating']} — {r['recommended_action']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Amount", f"₹{r['invoice_amount']:,.0f}")
    c2.metric("Days Overdue", r['days_overdue'])
    c3.metric("Vendor Tier", r['vendor_tier'])

    st.markdown("### ✉️ Drafted Communication — Awaiting Your Approval")
    st.markdown(r['drafted_communication'])

    st.warning("⏸️ Agent is PAUSED at the approval gate. State is persisted to disk. Nothing is sent until you approve.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve & Send", type="primary"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            with st.spinner("Resuming agent from checkpoint..."):
                app_graph.invoke(None, config)  # None = resume from pause
            st.session_state.phase = "done"
            st.rerun()
    with col2:
        if st.button("❌ Reject"):
            st.session_state.phase = "rejected"
            st.rerun()

# ─── PHASE 3a: DONE ───
elif st.session_state.phase == "done":
    st.success("✅ Approved. Agent resumed from checkpoint and communication was sent.")
    r = st.session_state.result
    st.write(f"**{r['vendor']}** — {r['recommended_action']} — dispatched.")
    if st.button("Process another invoice"):
        st.session_state.phase = "input"
        st.session_state.thread_id = None
        st.session_state.result = None
        st.rerun()

# ─── PHASE 3b: REJECTED ───
elif st.session_state.phase == "rejected":
    st.error("❌ Rejected. Communication was NOT sent. (In production: route to re-draft or manual handling.)")
    r = st.session_state.result
    st.write(f"**{r['vendor']}** — draft discarded, agent did not resume.")
    if st.button("Process another invoice"):
        st.session_state.phase = "input"
        st.session_state.thread_id = None
        st.session_state.result = None
        st.rerun()