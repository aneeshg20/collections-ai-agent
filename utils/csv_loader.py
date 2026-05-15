import pandas as pd
from pathlib import Path

REQUIRED_COLUMNS = {
    "invoice_id", "customer_id", "customer_name",
    "invoice_date", "due_date", "days_overdue",
    "invoice_amount", "outstanding_amount",
    "dispute_flag", "dispute_count", "avg_payment_delay_days",
    "payment_behaviour_score",
}


def load_invoices(filepath: str | Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["days_overdue"] = pd.to_numeric(df["days_overdue"], errors="coerce").fillna(0).astype(int)
    df["dispute_count"] = pd.to_numeric(df["dispute_count"], errors="coerce").fillna(0).astype(int)
    df["dispute_flag"] = df["dispute_flag"].astype(bool)
    df["outstanding_amount"] = pd.to_numeric(df["outstanding_amount"], errors="coerce").fillna(0.0)
    df["payment_behaviour_score"] = pd.to_numeric(df["payment_behaviour_score"], errors="coerce").fillna(5.0)

    return df[df["days_overdue"] > 0].reset_index(drop=True)
