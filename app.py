import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from utils.csv_loader import load_invoices
from utils.risk_rules import apply_rule_based_risk
from agent.classifier import classify_all
from agent.email_generator import draft_all_emails
from agent.telemetry import fetch_run_summary

st.set_page_config(page_title="Collections AI Agent", page_icon="💼", layout="wide")
st.title("💼 Collections AI Agent")
st.caption("Classify overdue invoices by dispute risk and generate personalised dunning emails.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    run_emails = st.toggle("Generate dunning emails", value=True)
    st.divider()
    st.subheader("Telemetry")
    if st.button("Refresh stats"):
        st.session_state["telemetry"] = fetch_run_summary()
    if "telemetry" in st.session_state:
        st.dataframe(pd.DataFrame(st.session_state["telemetry"]), use_container_width=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload overdue invoices CSV", type=["csv"])
if not uploaded:
    st.info("Upload a CSV to get started. See `data/sample_invoices.csv` for the expected schema.")
    st.stop()

# ── Load & pre-classify ───────────────────────────────────────────────────────
try:
    df = load_invoices(uploaded)
except ValueError as e:
    st.error(f"CSV validation failed: {e}")
    st.stop()

df = apply_rule_based_risk(df)
st.success(f"Loaded **{len(df)}** overdue invoices.")

with st.expander("Preview raw data"):
    st.dataframe(df, use_container_width=True)

# ── Classify ──────────────────────────────────────────────────────────────────
if st.button("Run AI Classification", type="primary"):
    with st.spinner("Classifying invoices with Claude..."):
        df = classify_all(df)
    st.session_state["classified_df"] = df

if "classified_df" not in st.session_state:
    st.stop()

df = st.session_state["classified_df"]

st.subheader("Classification Results")
col1, col2, col3 = st.columns(3)
counts = df["risk_tier"].value_counts()
col1.metric("🔴 High Risk", counts.get("High", 0))
col2.metric("🟡 Medium Risk", counts.get("Medium", 0))
col3.metric("🟢 Low Risk", counts.get("Low", 0))

tier_filter = st.selectbox("Filter by risk tier", ["All", "High", "Medium", "Low"])
view = df if tier_filter == "All" else df[df["risk_tier"] == tier_filter]
st.dataframe(view[["invoice_id", "customer_name", "days_overdue", "outstanding_amount",
                    "suggested_risk", "risk_tier", "reasoning"]], use_container_width=True)

# ── Email generation ──────────────────────────────────────────────────────────
if run_emails and st.button("Generate Dunning Emails"):
    with st.spinner("Drafting emails with Claude..."):
        df = draft_all_emails(df)
    st.session_state["emailed_df"] = df

if "emailed_df" in st.session_state:
    df = st.session_state["emailed_df"]
    st.subheader("Dunning Email Drafts")
    for _, row in df.iterrows():
        tier_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(row["risk_tier"], "")
        with st.expander(f"{tier_icon} {row['customer_name']} — Invoice {row['invoice_id']} ({row['risk_tier']})"):
            st.text_area("Email Draft", value=row["email_draft"], height=300,
                         key=f"email_{row['invoice_id']}")

    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download results as CSV", csv_out, "collections_results.csv", "text/csv")
