import sqlite3
import pandas as pd

# Read all telemetry
conn = sqlite3.connect("collections.db")
df = pd.read_sql("SELECT * FROM agent_runs", conn)
conn.close()

print(f"Total agent runs logged: {len(df)}")
print(f"\nColumns available: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df[['vendor', 'risk_rating', 'days_overdue', 'timestamp']].head())

print("\n" + "=" * 50)
print("RISK DISTRIBUTION")
print("=" * 50)

risk_counts = df['risk_rating'].value_counts()
total = len(df)
for rating, count in risk_counts.items():
    pct = (count / total) * 100
    print(f"{rating:<10} {count:>3} runs  ({pct:.1f}%)")

print("\n" + "=" * 50)
print("VENDOR RISK PATTERNS")
print("=" * 50)

# For each vendor, show how often they appear and their risk mix
vendor_risk = df.groupby('vendor')['risk_rating'].value_counts().unstack(fill_value=0)
print(vendor_risk)

print("\n--- Repeat HIGH-risk vendors (flagged 3+ times) ---")
high_risk_runs = df[df['risk_rating'] == 'HIGH']
repeat_offenders = high_risk_runs['vendor'].value_counts()
for vendor, count in repeat_offenders.items():
    if count >= 3:
        print(f"⚠️  {vendor}: flagged HIGH {count} times")


print("\n" + "=" * 50)
print("RECOMMENDED ACTION DISTRIBUTION")
print("=" * 50)

# Only analyze rows where recommended_action was captured
action_df = df[df['recommended_action'].notna()]

if len(action_df) == 0:
    print("No recommended_action data yet.")
    print("(Older runs predate this column - run new assessments to populate)")
else:
    action_counts = action_df['recommended_action'].value_counts()
    for action, count in action_counts.items():
        print(f"{action:<32} {count:>3}")