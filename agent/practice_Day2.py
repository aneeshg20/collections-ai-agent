invoice_id = "INV-001"
customer_name = "TCS"
invoice_amount = 25000
days_overdue = 45
dispute_flag = True

if days_overdue > 60:
    risk_tag = "HIGH"
elif days_overdue > 45 or dispute_flag:
    risk_tag = "MEDIUM"
else: risk_tag = "LOW"

print (f"Invoice amount: {invoice_amount}")
print (f"Risk Profile is {risk_tag}")
print (f"Dispute is {dispute_flag}")

# Day 2 - Lists, dictionaries, and loops
# Processing multiple invoices - batch running in SAP

invoices = [
    {"ID": "INV-001", "Customer":"TCS", "amount": 250000, "days_overdue": 91, "dispute": True},
    {"ID": "INV-002", "Customer":"McKinsey", "amount": 1250000, "days_overdue": 65, "dispute": False},
    {"ID": "INV-003", "Customer":"Bain", "amount": 2250000, "days_overdue": 20, "dispute": False},
    {"ID": "INV-004", "Customer":"BCG", "amount": 2000000, "days_overdue": 45, "dispute": True},
    {"ID": "INV-005", "Customer":"Kearney", "amount": 3000000, "days_overdue": 8, "dispute": False},
    {"ID": "INV-006", "Customer":"Infosys", "amount": 4100000, "days_overdue": 72, "dispute": True},
]

print ("\n" + "="*60)
print ("COLLECTIONS RISK REPORT")
print ("="*60)

high_risk = []
medium_risk = []
low_risk = []

for inv in invoices:
    if inv["days_overdue"] > 60:
        risk_tag = "High"
        high_risk.append(inv)
    elif inv["days_overdue"] > 30 or inv["dispute"]:
        risk_tag = "Medium"
        medium_risk.append(inv)
    else:
        risk_tag = "Low"
        low_risk.append(inv)
    print(f"{inv['ID']} | {inv['Customer']:<30} | {inv['amount']: > 10,} | {inv['days_overdue']: > 3} days | {risk_tag}")

print ("="*60)
print (f"High Risk: {len (high_risk)} invoices")
print (f"Medim Risk: {len (medium_risk)} invoices")
print (f"Low Risk: {len (low_risk)} invoices")
print (f"Total Amount at Risk (High): INR {sum(i['amount'] for i in high_risk):,}")
print (f"Total Invoices: {len(invoices)}")
print (f"Total high risk customers: {[i['Customer'] for i in high_risk]}")

for i in high_risk:
    print(f"{i['Customer']}")