import sqlite3
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()
client = Anthropic()

# Connect to database (creates the file if it doesn't exist)
conn = sqlite3.connect("collections.db")

result = pd.read_sql("SELECT * FROM invoices WHERE days_since_invoice > payment_term_days", conn)
print (result)

print (len(result))

for _, inv in result.iterrows():
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

    conn.close()