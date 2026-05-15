import json
import time
from pathlib import Path

import anthropic
import pandas as pd

from agent.telemetry import log_event

_client = anthropic.Anthropic()
_PROMPT_TEMPLATE = (Path(__file__).parent.parent / "prompts" / "classify.txt").read_text()
MODEL = "claude-sonnet-4-6"


def classify_invoice(row: pd.Series) -> dict:
    prompt = _PROMPT_TEMPLATE.format(
        invoice_id=row["invoice_id"],
        customer_id=row["customer_id"],
        customer_name=row["customer_name"],
        days_overdue=row["days_overdue"],
        outstanding_amount=row["outstanding_amount"],
        dispute_flag=row["dispute_flag"],
        dispute_count=row["dispute_count"],
        avg_payment_delay_days=row["avg_payment_delay_days"],
        payment_behaviour_score=row["payment_behaviour_score"],
        suggested_risk=row["suggested_risk"],
    )

    t0 = time.monotonic()
    response = _client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    content = response.content[0].text.strip()
    result = json.loads(content)

    log_event(
        invoice_id=str(row["invoice_id"]),
        customer_id=str(row["customer_id"]),
        event_type="classification",
        risk_tier=result["risk_tier"],
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=latency_ms,
        notes=result.get("reasoning", ""),
    )

    return result


def classify_all(df: pd.DataFrame) -> pd.DataFrame:
    results = df.apply(classify_invoice, axis=1, result_type="expand")
    df = df.copy()
    df["risk_tier"] = results["risk_tier"]
    df["reasoning"] = results["reasoning"]
    return df
