import time
from pathlib import Path

import anthropic
import pandas as pd

from agent.memory import retrieve_customer_context
from agent.telemetry import log_event

_client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

_PROMPTS = {
    "High":   (Path(__file__).parent.parent / "prompts" / "email_high.txt").read_text(),
    "Medium": (Path(__file__).parent.parent / "prompts" / "email_medium.txt").read_text(),
    "Low":    (Path(__file__).parent.parent / "prompts" / "email_low.txt").read_text(),
}


def draft_email(row: pd.Series) -> str:
    risk_tier = row["risk_tier"]
    template = _PROMPTS[risk_tier]
    customer_context = retrieve_customer_context(str(row["customer_id"]))

    prompt = template.format(
        invoice_id=row["invoice_id"],
        customer_name=row["customer_name"],
        outstanding_amount=row["outstanding_amount"],
        days_overdue=row["days_overdue"],
        dispute_count=row.get("dispute_count", 0),
        customer_context=customer_context or "No prior history on file.",
        reasoning=row.get("reasoning", ""),
    )

    t0 = time.monotonic()
    response = _client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    email_text = response.content[0].text.strip()

    log_event(
        invoice_id=str(row["invoice_id"]),
        customer_id=str(row["customer_id"]),
        event_type="email_draft",
        risk_tier=risk_tier,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=latency_ms,
    )

    return email_text


def draft_all_emails(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["email_draft"] = df.apply(draft_email, axis=1)
    return df
