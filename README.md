# Agentic LLM Pipeline for Natural Language to SQL

An agentic, schema-aware, multilingual **Natural Language to SQL (NL2SQL)** system that converts user questions into safe executable SQL for relational databases.

This project supports **text and voice input**, **multilingual normalization**, **schema-aware retrieval**, **query decomposition**, **iterative SQL refinement**, **result validation**, and **benchmark evaluation on Spider**. It is designed both as a practical prototype for database interaction and as a research framework for studying NL2SQL system behavior.

---

## Overview

Traditional Text-to-SQL systems often fail on complex queries due to weak schema grounding, poor reasoning over joins and aggregations, and lack of correction mechanisms. This project addresses those limitations through an **agentic multi-stage pipeline** instead of single-pass SQL generation.

The system supports:

- **Live database querying** on PostgreSQL and MySQL
- **Spider benchmark evaluation** on SQLite
- **CSV-based evaluation dashboards**
- **Voice-to-SQL workflow**
- **Multilingual input normalization to English**
- **Schema retrieval using vector search**
- **Structured query decomposition**
- **Safe SQL guardrails**
- **Execution, join-path, and result validation**
- **Iterative refinement when SQL fails**
- **Optional human-in-the-loop approval**
- **Learning from successful past queries**

---

## Key Features

### 1. Natural Language to SQL
Ask questions in plain language and generate SQL automatically.

Example:
- "Show total revenue by company in 2024"
- "List employees working in departments with more than 50 staff"

---

### 2. Multilingual Input Support
Typed questions can be entered in multiple languages and normalized into English before SQL generation.

Supported modes include:
- English
- Auto-detect
- Kannada
- Hindi
- Tamil
- Telugu
- Malayalam

---

### 3. Voice Input Pipeline
Users can record spoken questions, which are:
1. Transcribed
2. Language-detected
3. Translated to English
4. Sent into the NL2SQL pipeline

The system preserves both:
- original transcribed query
- translated English query

---

### 4. Dual Workflow Architecture

#### Baseline Workflow
A direct single-pass NL → SQL generation pipeline with semantic retries and DB retries.

#### Enhanced Phase 2 Workflow
An advanced pipeline with:
- schema retrieval
- decomposition planning
- SQL generation
- join validation
- result validation
- iterative refinement
- optional approval before execution

---

### 5. Schema-Aware Retrieval
Instead of sending the full schema to the LLM every time, the system:
- chunks the schema
- builds embeddings using Sentence Transformers
- stores vectors in FAISS
- retrieves top-k relevant schema chunks for a query

This improves:
- scalability
- relevance
- hallucination control

---

### 6. Structured Query Decomposition
A planning agent decomposes the question into:
- subquestions
- assumptions
- join hints
- required tables
- filters
- aggregations

This helps the model reason better about complex SQL generation.

---

### 7. SQL Safety Guardrails
Generated SQL is restricted to safe read-only queries.

The system blocks:
- INSERT
- UPDATE
- DELETE
- DROP
- ALTER
- multiple statements

Only safe **SELECT / CTE-based** queries are allowed.

---

### 8. Iterative Refinement
If SQL fails due to:
- syntax issues
- schema mismatch
- bad joins
- empty or suspicious results

the system enters a refinement loop and attempts to repair the SQL using feedback.

---

### 9. Validation Framework
The system validates generated SQL at multiple levels:

- **Execution validation** – does it run?
- **Schema validation** – are tables/columns valid?
- **Join-path validation** – do joins match FK relationships?
- **Result validation** – are results meaningful and plausible?

---

### 10. Human-in-the-Loop Support
An optional approval mode allows users to:
- review generated SQL
- edit SQL manually
- edit parameters
- approve before execution

This is useful in sensitive or production-style environments.

---

### 11. Learning from Past Queries
Successful NL → SQL pairs can be stored and reused as guidance for future generation.

This provides lightweight continual improvement without retraining the model.

---

### 12. Spider Benchmark Evaluation
The project includes evaluation utilities for the Spider dataset with:
- execution accuracy
- approximate exact match
- query hardness analysis
- join complexity analysis
- failure analysis
- enriched result CSV generation

---

### 13. Evaluation Dashboard
Evaluation outputs can be visualized through generated charts and dashboards such as:
- execution accuracy
- component accuracy
- hardness breakdown
- DB-wise accuracy
- join accuracy
- latency distribution
- outcome distribution

---

## Project Structure

```text
.
├── app.py                         # Main Streamlit application
├── nl2sql_runner.py              # Core NL2SQL pipeline orchestration
├── llm_factory.py                # LLM backend abstraction layer
├── prompts.py                    # Prompt templates and few-shot examples
├── phase2_agents.py              # Decomposer / generator / refiner agents
├── phase2_types.py               # Dataclasses for pipeline config and structures
├── input_utils.py                # Voice transcription and language normalization
├── db_utils.py                   # Database utility functions
├── schema_kb.py                  # Schema chunking, embedding, FAISS retrieval
├── schema_pack_utils.py          # Schema pack construction helpers
├── sql_checks.py                 # SQL safety checks
├── join_path_validator.py        # Join validation using FK graph
├── result_validator.py           # Output/result plausibility validation
├── tracing_utils.py              # Trace logging
├── spider_eval.py                # Spider benchmark evaluation runner
├── spider_metrics.py             # Evaluation metrics
├── spider_exact_match.py         # Exact/approximate match utilities
├── spider_failure_analysis.py    # Failure categorization
├── spider_component_analysis.py  # Component-level evaluation analysis
├── eval_dashboard_report.py      # Evaluation reporting
├── phase2_eval_dashboard.py      # Dashboard support
├── enrich_spider_csv.py          # Enrichment pipeline for eval CSVs
├── kb/                           # Schema knowledge base / FAISS indices
├── runs/                         # Trace logs from executions
├── sql/                          # Sample schemas and SQL scripts
├── tests/                        # Unit tests
└── requirements.txt              # Python dependencies