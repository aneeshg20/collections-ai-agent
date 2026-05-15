# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Collections AI Agent — an agentic Python/Streamlit app that reads overdue B2B invoice data, classifies each invoice as High/Medium/Low dispute risk using Claude, and drafts personalised dunning emails per risk tier. Domain: Accounts Receivable / Collections.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY
streamlit run app.py
```

## Common Commands

```bash
# Run the Streamlit app
streamlit run app.py

# Run rule-based unit tests (no API key needed)
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_classifier.py::test_high_risk_by_days -v
```

## Architecture

The agent pipeline runs in three stages:

1. **Ingest & pre-classify** (`utils/csv_loader.py` → `utils/risk_rules.py`)  
   CSV is validated, normalised, and each invoice gets a rule-based `suggested_risk` tier using domain heuristics (days overdue, dispute flag, payment behaviour score). This gives Claude a strong prior to confirm or override, reducing token usage.

2. **AI classification** (`agent/classifier.py`)  
   Calls Claude (`claude-sonnet-4-6`) with a structured prompt (`prompts/classify.txt`). Claude confirms or overrides the suggested tier and returns a JSON `{risk_tier, reasoning}`.

3. **Email generation** (`agent/email_generator.py`)  
   Retrieves customer history from ChromaDB (`agent/memory.py`), then calls Claude with a tier-specific prompt (`prompts/email_{high|medium|low}.txt`) to draft a personalised dunning email.

All Claude API calls are logged to SQLite via `agent/telemetry.py` (tokens, latency, risk tier, event type).

## Key Design Decisions

- **Two-stage classification**: rule-based pre-filter + Claude confirmation keeps latency and cost low. Override the rules in `utils/risk_rules.py` to tune tiers without touching prompts.
- **Prompt templates are plain text files** in `prompts/`. Edit them directly to tune tone, deadlines, or CTA language without touching Python.
- **ChromaDB memory** is used for customer-level context (payment notes, dispute history summaries). Seed it via `agent/memory.upsert_customer_note()` when loading the Kaggle/synthetic dataset.
- **Streamlit session state** holds `classified_df` and `emailed_df` so classification and email generation are independent steps — useful during prompt iteration.

## CSV Schema

Expected columns in the input file (see `data/sample_invoices.csv`):

| Column | Description |
|---|---|
| `invoice_id` | Unique invoice identifier |
| `customer_id` | Customer master ID |
| `customer_name` | Legal entity name |
| `days_overdue` | Integer, days past due date |
| `outstanding_amount` | Float, amount still unpaid |
| `dispute_flag` | Boolean, active dispute exists |
| `dispute_count` | Integer, historical dispute count |
| `avg_payment_delay_days` | Float, historical avg delay |
| `payment_behaviour_score` | Float 0–10, 10 = always pays on time |

## Environment Variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Required. Claude API access. |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path (default: `./db/chroma`) |
| `SQLITE_DB_PATH` | Telemetry DB path (default: `./db/telemetry.db`) |
