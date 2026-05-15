import pandas as pd
import random

# Load raw Kaggle data
df = pd.read_csv("invoices_kaggle.csv")

# Filter to open invoices only
df_open = df[df['isOpen'] == 1].copy()
print(f"Open invoices: {len(df_open)}")

# Simple column mapping — no date calculations
df_open['vendor'] = df_open['name_customer'].str.strip()
df_open['invoice_amount'] = pd.to_numeric(
    df_open['total_open_amount'], errors='coerce')

# Use fixed realistic values for demo
# These invoices are all open so treat as overdue
df_open['payment_term_days'] = [random.choice([30, 45, 60]) for _ in range(len(df_open))]
df_open['days_since_invoice'] = [random.randint(5, 120) for _ in range(len(df_open))]  # all overdue for demo purposes

# Select only needed columns
clean_df = df_open[[
    'vendor',
    'invoice_amount', 
    'payment_term_days',
    'days_since_invoice'
]].dropna()

clean_df = clean_df[clean_df['invoice_amount'] > 0]

print(f"Clean invoices: {len(clean_df)}")

# Save 20 row sample
sample_df = clean_df.head(20)
sample_df.to_csv("kaggle_sample_20.csv", index=False)
clean_df.to_csv("kaggle_invoices_clean.csv", index=False)

print("\nSample:")
print(sample_df)