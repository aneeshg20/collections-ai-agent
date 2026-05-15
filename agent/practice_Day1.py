invoice_id = "INV-001"
customer_name = "Tata Consultancy Services"
invoice_amount = 250000
days_overdue = 20
dispute_flag = False

print (f"Invoice: {invoice_id}")
print (f"Customer: {customer_name}")
print (f"Amount: INR {invoice_amount:,}")
print (f"Days Overdue: {days_overdue}")
print (f"In Dispute: {dispute_flag}")

if days_overdue > 60:
    risk_tier = "High"
elif days_overdue > 30 or dispute_flag:
    risk_tier = "Medium"
else:
    risk_tier = "Low"

print (f"Risk Tier: {risk_tier}")