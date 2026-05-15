import streamlit as st
import sqlite3
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="Collections AI Agent",
    page_icon="💰",
    layout="wide"
)

# Title
st.title("💰 Collections AI Agent")
st.subheader("Intelligent Invoice Risk Assessment")

# Connect to database
conn = sqlite3.connect("collections.db")

# Load invoice data
invoices = pd.read_sql("SELECT * FROM invoices", conn)

# Show invoice table
st.header("📋 Invoice Portfolio")
st.dataframe(invoices, use_container_width=True)

# Summary metrics
st.header("📊 Portfolio Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Invoices", len(invoices))

with col2:
    total_amount = invoices['invoice_amount'].sum()
    st.metric("Total Amount", f"${total_amount:,.0f}")

with col3:
    overdue = invoices[invoices['days_since_invoice'] > invoices['payment_term_days']]
    st.metric("Overdue Invoices", len(overdue))

conn.close()

st.header("📁 Upload new invoices")

uploaded_file = st.file_uploader("Upload a CSV file with invoice data", type="csv")

if uploaded_file is not None:
    # Read uploaded CSV
    new_data = pd.read_csv(uploaded_file)

    # Show preview
    st.subheader("Preview of uploaded data")
    st.dataframe(new_data,use_container_width=True)

    #Load into Database
    conn2 = sqlite3.connect("collections.db")
    new_data.to_sql("uploaded_invoices",conn2,if_exists="replace", index = False)
    new_data.to_sql("invoices", conn2, if_exists="replace", index=False)  # ← add this
    conn2.close()

    st.success(f"✅ {len(new_data)} invoices loaded successfully!")

    #Run Assessment button
    if st.button("🚀 Run Risk Assessment"):
        st.subheader("Running AI Risk Assessment")

        client=Anthropic()
        conn3 = sqlite3.connect("collections.db")
        invoices_to_assess = pd.read_sql("Select * from uploaded_invoices", conn3)

        results = []

        progress = st.progress(0)

        for i, (_,inv) in enumerate(invoices_to_assess.iterrows()):
            with st.spinner(f"Assessing {inv['vendor']}...."):
                message = client.messages.create(
                    model = "claude-sonnet-4-5",
                    max_tokens = 512,
                    temperature = 0,
                    system = "You are a collections risk analyst. Assess the invoice risk as HIGH, MEDIUM, or LOW. Start your response with exactly one of these words.",
                    messages = [{
                        "role": "user",
                        "content": f"Assess risk: {inv.to_dict()}"
                    }]
                )
                response = message.content[0].text
                if "HIGH" in response:
                    rating = "HIGH"
                elif "MEDIUM" in response:
                    rating = "MEDIUM"
                else:
                    rating = "LOW"

                results.append({
                    "vendor": inv['vendor'],
                    "invoice_amount": inv['invoice_amount'],
                    "risk_rating": rating,
                    "summary": response[:200]
                })

                progress.progress((i+1)/len(invoices_to_assess))

        conn3.close()
        # Define dataframe first
        results_df = pd.DataFrame(results)
        
        from datetime import datetime
        conn_log = sqlite3.connect("collections.db")
        for _, row in results_df.iterrows():
            conn_log.execute("""
                INSERT INTO agent_runs 
                (vendor, risk_rating, timestamp, tokens_used, model)
                VALUES (?, ?, ?, ?, ?)
            """, (
                row['vendor'],
                row['risk_rating'],
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                512,
                "claude-sonnet-4-5"
            ))
        conn_log.commit()
        conn_log.close()
        st.info("✅ Results saved to telemetry database")

        #Show results
        st.subheader("🎯 Risk Assessment Complete")
        
        def colour_risk(val):
            if val == "HIGH":
                return 'background-color: #ffcccc'
            elif val == "MEDIUM":
                return 'background-color: #fff3cc'
            elif val == "LOW":
                return 'background-color: #ccffcc'
            return ''
        
        st.dataframe(
            results_df.style.applymap(
                colour_risk, subset=['risk_rating']),
            use_container_width=True
        )

        # Risk summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 High Risk", 
                     len(results_df[results_df['risk_rating']=='HIGH']))
        with col2:
            st.metric("🟡 Medium Risk",
                     len(results_df[results_df['risk_rating']=='MEDIUM']))
        with col3:
            st.metric("🟢 Low Risk",
                     len(results_df[results_df['risk_rating']=='LOW']))


st.header("🎯 Risk Assessment Results")
conn4 = sqlite3.connect("collections.db")
risk_data = pd.read_sql("""
    SELECT i.vendor, i.invoice_amount, 
           a.risk_rating, a.timestamp
    FROM invoices i
    LEFT JOIN agent_runs a ON i.vendor = a.vendor
    ORDER BY a.timestamp DESC
""", conn4)
st.dataframe(risk_data, use_container_width=True)
conn4.close()

st.header("📈 Risk Distribution")
# Filter out null risk ratings for the chart
chart_data = risk_data.dropna(subset=['risk_rating'])
risk_counts = chart_data['risk_rating'].value_counts()
st.bar_chart(risk_counts)

def colour_risk(val):
    if val == "HIGH":
        return 'background-color: #ffcccc'
    elif val == "MEDIUM":
        return 'background-color: #fff3cc'
    elif val == "LOW":
        return 'background-color: #ccffcc'
    return ''

st.dataframe(
    risk_data.style.applymap(colour_risk, subset=['risk_rating']),
    use_container_width=True
)


