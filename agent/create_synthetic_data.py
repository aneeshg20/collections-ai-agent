import pandas as pd
from datetime import datetime, timedelta

# ============================================================
# create_synthetic_data.py
# Generates 5 synthetic CSV files for Collections AI Agent UAs
# Honeywell = HIGH risk, Siemens = LOW risk, others = mixed
# ============================================================

vendors = ["Siemens", "Honeywell", "ABB", "Bosch", "Schneider"]

# ── 1. PTP HISTORY ──────────────────────────────────────────
ptp_data = {
    "vendor":           ["Siemens",    "Siemens",    "Honeywell",  "Honeywell",  "Honeywell",  "ABB",        "ABB",        "Bosch",      "Bosch",      "Schneider"],
    "promise_date":     ["2026-01-10", "2026-02-15", "2026-01-05", "2026-02-10", "2026-03-15", "2026-01-20", "2026-02-25", "2026-01-15", "2026-02-20", "2026-02-10"],
    "promised_amount":  [500000,       750000,       1200000,      980000,       1100000,      750000,       620000,       320000,       290000,       890000],
    "actual_pay_date":  ["2026-01-09", "2026-02-14", "2026-01-25", "2026-03-05", "2026-04-10", "2026-01-22", "2026-03-10", "2026-01-18", "2026-02-19", "2026-02-25"],
    "broken_flag":      [False,        False,        True,         True,         True,         False,        True,         False,        False,        False],
}
df_ptp = pd.DataFrame(ptp_data)
df_ptp.to_csv("agent/ptp_history.csv", index=False)
print("✅ ptp_history.csv created")
print(df_ptp.to_string(index=False))

# ── 2. DISPUTE HISTORY ──────────────────────────────────────
dispute_data = {
    "vendor":         ["Honeywell",  "Honeywell",  "ABB",        "Bosch",      "Schneider",  "Siemens"],
    "dispute_id":     ["D001",       "D002",       "D003",       "D004",       "D005",       "D006"],
    "dispute_type":   ["Pricing",    "Quality",    "Delivery",   "Duplicate Invoice", "Pricing", "Quality"],
    "amount":         [250000,       180000,       95000,        320000,       140000,       60000],
    "status":         ["Open",       "Open",       "Pending",    "Resolved",   "Pending",    "Resolved"],
    "raised_date":    ["2026-03-10", "2026-04-01", "2026-02-15", "2026-01-20", "2026-03-25", "2026-01-05"],
}
df_disputes = pd.DataFrame(dispute_data)
df_disputes.to_csv("agent/dispute_history.csv", index=False)
print("\n✅ dispute_history.csv created")
print(df_disputes.to_string(index=False))

# ── 3. CREDIT BALANCE ───────────────────────────────────────
credit_data = {
    "vendor":             ["Siemens",  "Honeywell", "ABB",     "Bosch",   "Schneider"],
    "credit_limit":       [2000000,    2000000,     1500000,   1000000,   1500000],
    "current_exposure":   [600000,     1900000,     900000,    650000,    1200000],
    "utilisation_pct":    [30.0,       95.0,        60.0,      65.0,      80.0],
}
df_credit = pd.DataFrame(credit_data)
df_credit.to_csv("agent/credit_balance.csv", index=False)
print("\n✅ credit_balance.csv created")
print(df_credit.to_string(index=False))

# ── 4. VENDOR MASTER ────────────────────────────────────────
vendor_master_data = {
    "vendor":              ["Siemens",   "Honeywell", "ABB",       "Bosch",     "Schneider"],
    "tier":                ["Strategic", "At-Risk",   "Preferred", "Preferred", "Standard"],
    "payment_score":       [85,          35,          65,          70,          58],
    "avg_days_to_pay":     [28,          67,          42,          32,          44],
    "country":             ["Germany",   "USA",       "Switzerland","Germany",  "France"],
    "relationship_years":  [8,           5,           6,           10,          4],
}
df_vendor = pd.DataFrame(vendor_master_data)
df_vendor.to_csv("agent/vendor_master.csv", index=False)
print("\n✅ vendor_master.csv created")
print(df_vendor.to_string(index=False))

# ── 5. AGING BUCKETS ────────────────────────────────────────
aging_data = {
    "vendor":         ["Siemens",  "Honeywell", "ABB",    "Bosch",   "Schneider"],
    "bucket_0_30":    [500000,     0,           750000,   320000,    890000],
    "bucket_31_60":   [0,          0,           0,        0,         0],
    "bucket_61_90":   [0,          1200000,     0,        0,         0],
    "bucket_90_plus": [0,          0,           0,        0,         0],
}
df_aging = pd.DataFrame(aging_data)
df_aging.to_csv("agent/aging_buckets.csv", index=False)
print("\n✅ aging_buckets.csv created")
print(df_aging.to_string(index=False))

print("\n🎉 All 5 synthetic datasets created successfully!")
print("Files saved to agent/ folder:")
print("  → ptp_history.csv")
print("  → dispute_history.csv")
print("  → credit_balance.csv")
print("  → vendor_master.csv")
print("  → aging_buckets.csv")
