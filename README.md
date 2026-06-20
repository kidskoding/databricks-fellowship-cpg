# CPG Trade Promotion Agent — Mosaic AI on Azure Databricks

A small, end-to-end demo that builds a **conversational analytics agent** for the
Consumer Packaged Goods (CPG) industry. The agent answers questions about
**promotion performance and trade-spend effectiveness** by querying real retail
transaction data — and every step is governed by Unity Catalog and traced in MLflow.

Think: *"Which departments have the highest promo lift, and where should Pepsi
focus trade spend next quarter?"* → the agent runs SQL against governed Delta
tables, reasons over the numbers, and gives a concrete recommendation.

---

## What this project actually does

There are two Databricks notebooks. You run them in order.

1. **`notebooks/01_setup.py`** — Data ingestion
   - Downloads the [Dunnhumby "The Complete Journey"](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey)
     retail dataset from Kaggle (via `kagglehub`).
   - Writes three CSVs into **Delta tables** under the Unity Catalog schema
     `databricks_cpg.cpg_demo`:
     - `transactions` — line-item sales (household, product, week, sales value)
     - `causal` — promotion exposure (in-store `display`, `mailer` flags)
     - `products` — product → department/commodity hierarchy

2. **`notebooks/02_agent.py`** — The agent
   - Builds a **tool-calling LangChain agent** powered by a Databricks-hosted LLM
     (`databricks-meta-llama-3-3-70b-instruct` via `ChatDatabricks`).
   - Exposes three tools, each a governed Spark SQL query over the Delta tables:
     - `get_top_departments()` — list available CPG departments
     - `get_promo_lift(department)` — promo avg sale vs. baseline avg sale + % lift
     - `get_weekly_promo_trend(department)` — week-over-week sales, flagged by promo
   - Wraps everything in **MLflow autolog** (`mlflow.langchain.autolog()`), so each
     run captures the full reasoning trace, tool calls, inputs, and outputs under
     the experiment `/cpg-promo-agent`. No black box.

---

## The big idea

This is a reference pattern for **trustworthy, governed AI agents on Databricks**:

- **Unity Catalog** = governance. The agent only touches data it is allowed to,
  through named catalog/schema/tables — not raw files.
- **Delta tables** = the queryable, versioned source of truth.
- **MLflow tracing** = auditability. Every agent decision is logged and replayable.
- **Tool-calling LLM** = the agent computes answers with real SQL instead of
  hallucinating numbers. It must cite specific figures and recommend a next action.

The business framing is CPG commercial analytics (Pepsi, P&G, etc.): measuring
whether trade-promotion spend (displays, mailers) actually lifts sales.

---

## Architecture at a glance

```
Kaggle (Dunnhumby)
      │  01_setup.py (kagglehub download)
      ▼
Delta tables  ──  databricks_cpg.cpg_demo.{transactions, causal, products}
      │              (governed by Unity Catalog)
      ▼
@tool functions (Spark SQL)  ──  get_top_departments / get_promo_lift / get_weekly_promo_trend
      │
      ▼
LangChain tool-calling agent  +  ChatDatabricks (Llama 3.3 70B)
      │
      ▼
MLflow trace  ──  experiment /cpg-promo-agent  (full reasoning + tool calls logged)
```

---

## Running it

These notebooks are meant to run **inside an Azure Databricks workspace**, not
locally. They rely on Databricks runtime globals (`spark`, `dbutils`) and
Databricks model serving endpoints.

### Prerequisites
- An Azure Databricks workspace with **Unity Catalog** enabled.
- A catalog named `databricks_cpg` (or edit the notebooks to match yours).
- Access to the `databricks-meta-llama-3-3-70b-instruct` serving endpoint.
- Kaggle credentials configured for `kagglehub` (the dataset is public).

### Steps
1. Import both files in `notebooks/` into your workspace (they're in Databricks
   `# COMMAND ----------` notebook source format).
2. Run **`01_setup.py`** top to bottom on a cluster. Confirm the three tables
   appear via the final `SHOW TABLES` cell.
3. Run **`02_agent.py`**. The last cell invokes the agent with a sample question;
   the answer prints inline and the full trace lands in the MLflow experiment.
4. Ask your own questions by changing the `"input"` string in the final cell.

---

## Repo layout

```
.
├── notebooks/
│   ├── 01_setup.py     # download dataset → Delta tables in Unity Catalog
│   └── 02_agent.py     # LangChain tool-calling agent + MLflow tracing
├── pyproject.toml      # local dev deps (uv) — mainly kagglehub
├── uv.lock
└── README.md
```

## Local dev (optional)

The `pyproject.toml` / `uv.lock` exist mainly for local experimentation with
`kagglehub` (e.g. inspecting the dataset before uploading). The agent runtime
deps (`databricks-langchain`, `mlflow`, `langchain`, `langgraph`) are installed
**inside the notebook** via `%pip install`, so there's no full local environment
to reproduce the agent. To poke at the dataset locally:

```bash
uv sync
```

---

## Notes & caveats

- **SQL injection:** the tools interpolate `department` directly into SQL. Fine
  for a trusted demo; parameterize before any real-world use.
- **Catalog name** `databricks_cpg` is hardcoded — change it everywhere if your
  workspace differs.
- Promotion detection is heuristic: a row counts as "on promo" when `display` or
  `mailer` is non-`'0'` in the `causal` table.
