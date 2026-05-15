from anthropic import Anthropic
from dotenv import load_dotenv
import os

# API key removed - use environment variable

invoices_record = [
    {"Vendor": "Tech Mahindra","Invoice_Amount": 250000, "days_overdue": 25, "dispute_flag": True, "payment_terms": 60},
]

load_dotenv()
client = Anthropic()

message = client.messages.create(
    model = "claude-sonnet-4-5",
    max_tokens=1024,
    system = "You are an agent in a world class collections team for a managed services account in a top tier consultancy firm. Your job is to idetify the collection risk in terms of breaking promise to pay, possible delinquency, impact in DSO and working capital. You must also raise a flag if its going into a bad debt zone by checking customer payment history and trend as well. Give the response as High, Medium and Low and also your reasoning for the same",
    messages=[
        {"role":"user", "content": f" Assess collection risk for this invoice: {invoices_record[0]}"}
    ]
)

print(message.content[0].text)

