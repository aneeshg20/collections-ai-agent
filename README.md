# 💰 Collections AI Agent

An intelligent invoice risk assessment agent built with Python, Claude AI, Streamlit, and SQLite. Designed to automate collections strategy decisions for enterprise AR portfolios.

## 🎯 What it does

- Reads accounts receivable invoice data (CSV upload or database)
- Classifies each invoice as **HIGH**, **MEDIUM**, or **LOW** collection risk using Claude AI
- Generates structured risk assessments with reasoning
- Logs every agent run to a telemetry database for auditability
- Displays results in an interactive Streamlit dashboard

## 🏗️ Architecture

Invoice Data (CSV / SQLite)
↓
Risk Classification Engine (rule-based pre-filter)
↓
Claude API (claude-sonnet-4-5, temperature=0)
↓
Risk Rating Extraction (HIGH / MEDIUM / LOW)
↓
Telemetry Logger (SQLite agent_runs table)
↓
Streamlit Dashboard (colour-coded results + charts)

## 📊 Domain Context

Built on real collections process knowledge across:
- Accounts Receivable (AR) and Dunning workflows  
- AP / Procure-to-Pay (PTP) process experience
- DSO, Working Capital, and Bad Debt risk metrics
- Enterprise collections strategy for managed services accounts

## 🗄️ Dataset

- **Primary**: [B2B Invoice Payment Prediction Dataset](https://www.kaggle.com/datasets/pradumn203/payment-date-prediction-for-invoices-dataset) — 50,000 real enterprise invoices (Kaggle)
- **Supplementary**: Synthetic AR data generated to simulate Indian enterprise context

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| AI/LLM | Claude Sonnet 4.5 (Anthropic API) |
| UI | Streamlit |
| Database | SQLite (invoices + telemetry) |
| Data | Pandas |
| Language | Python 3.12 |

## 🚀 Run locally

**1. Clone the repo:**
```bash
git clone https://github.com/YOUR_USERNAME/collections-agent.git
cd collections-agent
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Set up environment:**
```bash
cp .env.example .env
# Add your Anthropic API key to .env
```

**4. Initialise the database:**
```bash
python setup_database.py
```

**5. Run the app:**
```bash
streamlit run app.py
```

## 📁 Project Structure

collections-agent/
├── app.py                    # Streamlit dashboard
├── setup_database.py         # Database initialisation
├── telemetry_logger.py       # Agent pipeline with logging
├── clean_kaggle_data.py      # Kaggle data cleaning script
├── practice_sql1.py          # SQL JOIN queries
├── data/
│   ├── invoices.csv          # Sample invoice data
│   └── kaggle_sample_20.csv  # Cleaned Kaggle sample
├── requirements.txt
├── .env.example
└── README.md

## 🔮 Roadmap

- [ ] LangGraph multi-agent orchestration
- [ ] ChromaDB RAG memory layer
- [ ] Dunning email drafting agent
- [ ] SOA (Statement of Accounts) generation
- [ ] Docker containerisation
- [ ] Streamlit Cloud deployment

## 👤 Author

IIT Kanpur MBA | Operations & Analytics  
Domain:  AP/PTP · Supply Chain · Collections · Process Excellence · Program Management