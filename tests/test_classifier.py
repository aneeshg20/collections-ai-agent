import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from utils.risk_rules import rule_based_risk


def _make_row(**kwargs) -> pd.Series:
    defaults = dict(
        invoice_id="INV-TEST", customer_id="C-001", customer_name="Test Co",
        days_overdue=0, outstanding_amount=10000, dispute_flag=False,
        dispute_count=0, avg_payment_delay_days=5, payment_behaviour_score=8.0,
    )
    return pd.Series({**defaults, **kwargs})


TESTS = [
    ("INV-001", dict(days_overdue=91),                                   "High",   "91 days overdue"),
    ("INV-002", dict(days_overdue=65, dispute_flag=True),                "High",   "65 days + active dispute"),
    ("INV-003", dict(days_overdue=35, dispute_count=3, dispute_flag=True),"High",  "31+ days with 3 disputes"),
    ("INV-004", dict(days_overdue=45),                                   "Medium", "45 days overdue, no dispute"),
    ("INV-005", dict(days_overdue=10, payment_behaviour_score=3.0),      "Medium", "poor pay score (<4)"),
    ("INV-006", dict(days_overdue=16, dispute_flag=True),                "Medium", "14+ days + active dispute"),
    ("INV-007", dict(days_overdue=5),                                    "Low",    "only 5 days overdue"),
]


def run_tests():
    passed = 0
    failed = 0

    print("\n" + "=" * 65)
    print(f"  Collections Agent — Risk Rule Tests")
    print("=" * 65)

    for invoice_id, overrides, expected_tier, description in TESTS:
        row = _make_row(invoice_id=invoice_id, **overrides)
        actual = rule_based_risk(row)
        ok = actual == expected_tier
        status = "PASS" if ok else "FAIL"

        print(f"\n  [{status}] {invoice_id}")
        print(f"         Scenario : {description}")
        print(f"         Expected : {expected_tier}")
        print(f"         Got      : {actual}")

        if ok:
            passed += 1
        else:
            failed += 1
            print(f"         !! Mismatch — check risk_rules.py thresholds")

    print("\n" + "-" * 65)
    print(f"  Results: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    print("=" * 65 + "\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
