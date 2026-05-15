from anthropic import Anthropic
from dotenv import load_dotenv
import os

invoices_record = [
    {"Vendor": "Tech Mahindra","Invoice_Amount": 250000, "days_overdue": 25, "dispute_flag": True, "payment_terms": 60},
    {"Vendor": "Maruti Suzuki","Invoice_Amount": 650000, "days_overdue": 5, "dispute_flag": False, "payment_terms": 30},
    {"Vendor": "D-Mart","Invoice_Amount": 150000, "days_overdue":15, "dispute_flag": True, "payment_terms": 15}
]

load_dotenv()
client = Anthropic()

for inv in invoices_record:
    message = client.messages.create(
    model = "claude-sonnet-4-5",
    max_tokens=1024,
    system = "You are an agent in a world class collections team for a managed services account in a top tier consultancy firm. Your job is to idetify the collection risk in terms of breaking promise to pay, possible delinquency, impact in DSO and working capital. You must also raise a flag if its going into a bad debt zone by checking customer payment history and trend as well. Give the response as High, Medium and Low and also your reasoning for the same",
    messages=[
        {"role":"user", "content": f" Assess collection risk for this invoice: {inv}"}
             ]
    )
    print(message.content[0].text)
    print("\n" + "="*60 + "\n")

print (f"Total invoices processed: {len(invoices_record)}")
