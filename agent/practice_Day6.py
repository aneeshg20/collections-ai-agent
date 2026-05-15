import pandas as pd
import numpy as np

data = pd.read_csv("invoices.csv")
print (data.head())

for index, row in data.iterrows():
    vendor = row["vendor"]
    terms = row["payment_term_days"]
    days = row["days_since_invoice"]

    if days < terms:
        status = "ON TIME"
    elif days == terms:
        status = "DUE TODAY"
    else:
        status = "OVERDUE"

    print (f" {vendor: <12} | Terms: {terms} days | Status: {status}")

print ("\n" + "="*50)
print (f"Total vendors: {len(data)}")

On_time_vendors = []
overdue_vendors = []

for index, row in data.iterrows():
    vendor = row["vendor"]
    terms = row["payment_term_days"]
    days = row["days_since_invoice"]

    if days < terms:
        On_time_vendors.append(vendor)
    else:
        overdue_vendors.append(vendor)

print (f"on time vendors: {len(On_time_vendors)}")
print (f"overdue vendors: {len(overdue_vendors)}")
print (f"total amount overdue: {data[data['vendor'].isin(overdue_vendors)]['invoice_amount'].sum()}")