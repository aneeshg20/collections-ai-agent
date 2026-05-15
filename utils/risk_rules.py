"""
Rule-based pre-classifier. Assigns a suggested risk tier before Claude refines it.
Reduces API calls by giving Claude a strong prior to confirm or override.

Scoring logic (domain heuristics):
  - Days overdue heavily weighted (>90 = High, 31-90 = Medium, <=30 = Low baseline)
  - Active dispute bumps risk up one tier
  - Poor payment behaviour score (<4 out of 10) bumps risk up one tier
"""

import pandas as pd


def rule_based_risk(row: pd.Series) -> str:
    days = int(row["days_overdue"])
    has_dispute = bool(row["dispute_flag"])
    dispute_count = int(row["dispute_count"])
    pay_score = float(row["payment_behaviour_score"])

    if days > 90 or (days > 60 and has_dispute) or (days > 30 and dispute_count >= 3):
        tier = "High"
    elif days > 30 or (days > 14 and has_dispute) or pay_score < 4:
        tier = "Medium"
    else:
        tier = "Low"

    return tier


def apply_rule_based_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["suggested_risk"] = df.apply(rule_based_risk, axis=1)
    return df
