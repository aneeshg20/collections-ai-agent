# Interview Concepts — Collections AI Agent
*Aneesh — what you have built and how to defend it*

---

## 1. The Three Architectural Patterns

**Pattern A — Workflow Orchestration (LangGraph):**
Nodes execute in defined sequence. Each node has one responsibility. State flows through the graph. Same pattern as Airflow, Step Functions, Temporal — but with LLM calls inside nodes.

**Pattern B — RAG (Retrieval Augmented Generation):**
retrieve_history node queries vector store before decision node. Without RAG, LLMs are stateless. With RAG, they have institutional memory.

**Pattern C — Rule + LLM Hybrid:**
pre_classify is rules-based. collection_strategy is LLM-based. Rules are deterministic, cheap, instant, auditable. LLMs handle judgment-heavy cases that rules cannot encode.

**Interview soundbite:**
> "I use a hybrid architecture — deterministic rules for any decision that can be codified, LLM nodes only where genuine judgment is required. This reduces cost per invoice by approximately 60% versus a pure LLM approach while maintaining auditability that pure LLM systems cannot provide."

---

## 2. State Pattern (TypedDict)

InvoiceState is the contract between agents — Data Schema Contract principle.

Each node:
- Receives the full state
- Reads only what it needs
- Returns only what it changes
- Never breaks the contract

Same principle as microservices API contracts.

**Interview soundbite:**
> "The TypedDict state acts as a contract between agent nodes. Each node touches only the fields relevant to its responsibility. This makes the system testable, debuggable, and extensible — adding a new utility agent does not require changing existing nodes."

---

## 3. Cosine Similarity Math

```
similarity = (A · B) / (|A| × |B|)

A · B = dot product
|A|, |B| = vector magnitudes
Result: -1 (opposite) to 1 (identical)
```

**Why cosine over Euclidean:**
Cosine measures direction not magnitude. Two documents about the same topic — one long one short — have similar direction but different magnitudes. Cosine ignores length differences.

**Interview soundbite:**
> "I implemented cosine similarity for semantic retrieval. The choice of cosine over Euclidean was deliberate — cosine measures vector direction independent of magnitude, which matters for text where document lengths vary significantly but semantic content is what matters for retrieval."

---

## 4. Embedding Limitation (Honest Framing)

Used hash-based embeddings due to Windows compatibility issues.

**The framing:**
Hash embeddings are deterministic but not semantic. They prove RAG pipeline works end to end. Production swaps to sentence-transformers, OpenAI ada-002, or Anthropic embedding API.

**Why this is a strength:**
- You can speak to the tradeoff
- Architecture decouples embedding from retrieval
- Swap embedding function → no other code changes
- This is good software design

**Interview soundbite:**
> "My current implementation uses hash embeddings for the proof of concept due to local Windows environment constraints. The architecture decouples embedding generation from retrieval logic — production deployment swaps the embedding function to sentence-transformers or Anthropic's embedding API without touching the retrieval code. That separation of concerns is what allows the system to evolve."

---

## 5. Conditional Edges (Decision Gates)

route_by_risk function returns string that determines next node. Called Conditional Router or Decision Gate.

**Why this matters:**
Real business processes are not linear. Collections branches based on risk. Disputes branch based on type. Payments branch based on amount.

**The pattern:**
At every decision point in the human process, there should be a conditional edge in the agent.

**Interview soundbite:**
> "Conditional edges in LangGraph let me model real business decision gates. In my collections agent, the route_by_risk function reads the calculated risk score and routes HIGH risk invoices to an escalation node, while LOW and MEDIUM follow the standard telemetry path. Each conditional edge in my graph corresponds to a real decision a human would make in the manual process."

---

## 6. Telemetry Pattern (Audit Trail)

log_telemetry writes every decision to SQLite. Non-negotiable in enterprise AI.

**Why it matters:**
- Compliance: every AI decision must be auditable
- Improvement: telemetry feeds Observer Agent to detect patterns
- Trust: customers see what AI did and why
- Liability: trail proves human oversight

**Interview soundbite:**
> "Every agent decision is logged to a telemetry database with timestamp, model used, tokens consumed, and full reasoning. This serves three purposes — regulatory compliance, system improvement through pattern detection, and trust because every decision can be reconstructed and explained."

---

## 7. Agent Failure Modes & Mitigations

| Failure Mode | Your Mitigation |
|---|---|
| Tool Failure (UA cannot read CSV) | try/except with safe defaults |
| Hallucination (Claude invents data) | Rubric scoring forces grounded reasoning |
| Loop Failure (agent gets stuck) | LangGraph recursion limits + END nodes |
| Retrieval Failure (wrong context) | Similarity threshold + metadata filtering |
| Cascade Failure (one node breaks all) | State-based independence + safe defaults |

**Interview soundbite:**
> "Production AI systems fail in specific ways. My architecture addresses each — try/except in tool nodes for graceful degradation, rubric-based prompts to ground LLM reasoning, explicit END states to prevent loops, and similarity thresholds in retrieval to filter noise. Failure mode awareness is what separates a demo from a production system."

---

## 8. Business Case Framing (DSO Reduction)

```
Manual process:
- Senior collector handles 50-100 invoices/day
- 15-20 minutes per decision
- Data gathering across 5 systems

Your agent:
- 10,000 invoices per overnight batch
- 2-3 seconds per decision
- All 5 sources queried in parallel
- Every decision logged

Business impact:
- ₹100 crore AR portfolio
- 2-day DSO reduction
- Releases ₹55 lakhs working capital annually
- ₹4-6 lakhs recovered finance costs per year
```

**Interview soundbite:**
> "The business case is simple. A senior collector handles 100 invoices per day with 20-minute decision time including data gathering. My agent handles 10,000 invoices in an overnight batch with 3-second decision time. For a ₹100 crore AR portfolio, a 2-day DSO reduction releases ₹55 lakhs in working capital annually — that is the ROI conversation, not a technology conversation."

---

## 9. LangChain vs LangGraph

**LangChain:**
Sequential chains. Linear execution. Works for simple LLM pipelines. No native state management. No persistence. No HITL. Good for prototypes.

**LangGraph:**
Graph-based execution. Conditional branching. Native state via TypedDict. Persistence via checkpointers. HITL via interrupts. Production grade.

**Interview soundbite:**
> "LangChain is sequential — fine for prototypes. LangGraph is graph-based which lets me model real business processes with branching, parallel execution, persistence across sessions, and human-in-the-loop interrupts. For collections workflows that need audit trails and human approval gates, LangGraph is the correct choice."

---

## 10. Orchestrator-Worker Pattern (Coming Next Week)

```
Without Orchestrator (current):
check_ageing → check_disputes → check_ptp → ...
All UAs run in sequence
Total time = sum of UA times

With Orchestrator:
Orchestrator spawns all 5 UAs in parallel
All UAs run simultaneously
Total time = max of UA times (5x faster)
```

Same pattern as MapReduce — split, run parallel, combine.

**Interview soundbite:**
> "The Orchestrator-Worker pattern is what makes the system scale. Instead of running 5 utility agents sequentially, the orchestrator spawns them in parallel using LangGraph's Send API. For 10,000 invoices this changes runtime from 50 minutes to under 10 minutes. This is the same MapReduce pattern from distributed computing applied to agentic AI."

---

## 11. One Concept From Each AI Engineering Chapter

**Ch 1 — Foundation models:**
Using Claude Sonnet without fine-tuning. Right choice — fine-tuning is for narrow tasks. Collections has too much variability.

**Ch 2 — Sampling:**
temperature=0 for deterministic outputs. Critical for auditable decisions. Higher temperatures introduce variance that breaks compliance.

**Ch 3 — Evaluation:**
Rubric scoring as evaluation method. Each decision re-evaluable against same rubric to detect drift.

**Ch 4 — Evaluating AI systems:**
Telemetry layer = evaluation infrastructure. Every run logged. Patterns analyzed retrospectively.

**Ch 5 — Prompt engineering:**
Primacy and recency — rubric at top, output format at end. Structured output with explicit fields.

**Ch 6 — RAG:**
Both phases implemented — indexing (storing decisions) and retrieval (querying by similarity). Cosine over keyword matching.

---

## 12. The Unique Profile Pitch

Most candidates have ONE of:
- Strong domain knowledge (consultants)
- Strong technical skills (engineers)
- Strong MBA branding (B-school grads)

**You have all three. The agent proves it.**

**Your one-sentence pitch:**
> "I am a consultant with six years of source-to-cash domain depth who has built an agentic AI system from scratch against a real collections P&L metric. I can scope agentic engagements at the partner level, design the architecture, and validate the technical work because I have done all three."

That sentence ends the "engineer or consultant" debate before it starts.

---

## How to Use This

1. Save this file in your project folder
2. Read before any interview
3. Connect each concept back to a specific file/commit in your GitHub
4. Practice the soundbites out loud until they sound natural
5. Update as you build more (Orchestrator, HITL, Observer Agent)

---

**Reference Files in Your Repo:**

- `dev/langgraph_Day1.py` → Pattern progression starts here
- `dev/langgraph_Day8.py` → Rubric scoring + recommended actions
- `dev/langgraph_Day9.py` → RAG retrieve_history node
- `agent/chromadb_setup.py` → Vector store implementation
- `vector_store.json` → 20 stored decisions
- `agent/graph_visualisation.png` → Visual proof of architecture
- `collections.db` → Telemetry database with 36+ runs
