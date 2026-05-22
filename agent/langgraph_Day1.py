from anthropic import Anthropic
from dotenv import load_dotenv
from datetime import datetime
import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

load_dotenv()
client = Anthropic()


invoice = {
    "vendor": "Honeywell",
    "invoice_amount": 1200000,
    "days_since_invoice": 67,
    "payment_term_days": 45,
    "risk_rating": "",
    "reasoning": ""
}


class InvoiceState (TypedDict):
    vendor: str
    invoice_amount: float
    days_since_invoice: int
    payment_term_days: int
    risk_rating: str
    reasoning: str

def assess_risk(state: InvoiceState):
    message = client.messages.create(
    model = "claude-sonnet-4-5",
    max_tokens=1024,
    system = "You are an agent in a world class collections team for a managed services account in a top tier consultancy firm. Your job is to idetify the collection risk in terms of breaking promise to pay, possible delinquency, impact in DSO and working capital. You must also raise a flag if its going into a bad debt zone by checking customer payment history and trend as well. Give the response as High, Medium and Low and also your reasoning for the same",
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

# Build the graph
graph = StateGraph(InvoiceState)
graph.add_node("assess_risk", assess_risk)
graph.add_edge(START, "assess_risk")
graph.add_edge("assess_risk", END)
app = graph.compile()

# Run it
result = app.invoke(invoice)
print(f"Vendor: {result['vendor']}")
print(f"Risk Rating: {result['risk_rating']}")
print(f"Reasoning: {result['reasoning'][:200]}")