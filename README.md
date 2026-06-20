# CPG Trade Promotion Agent — Mosaic AI on Azure Databricks

A small, end-to-end demo that builds a **conversational analytics agent** for the
Consumer Packaged Goods (CPG) industry. The agent answers questions about
**promotion performance and trade-spend effectiveness** by querying real retail
transaction data — and every step is governed by Unity Catalog and traced in MLflow.

Think: *"Which departments have the highest promo lift, and where should Pepsi
focus trade spend next quarter?"* → the agent runs SQL against governed Delta
tables, reasons over the numbers, and gives a concrete recommendation.

---

## Quickstart

> Runs **inside an Azure Databricks workspace** (needs Unity Catalog + the
> `databricks-meta-llama-3-3-70b-instruct` serving endpoint). It is not a local app.

1. **Import** `notebooks/01_setup.py` and `notebooks/02_agent.py` into your
   workspace (Databricks notebook source format).
2. **Setup** — run `01_setup.py` top to bottom. It downloads the Dunnhumby
   dataset from Kaggle and writes three Delta tables into
   `databricks_cpg.cpg_demo`. The final `SHOW TABLES` cell should list
   `transactions`, `causal`, `products`.
3. **Run the agent** — run `02_agent.py`. The last cell asks a sample question and
   prints the answer inline; the full reasoning trace lands in the MLflow
   experiment `/cpg-promo-agent`.
4. **Ask your own** — edit the `"input"` string in the final cell, e.g.
   *"What's the promo lift for the GROCERY department?"*

> Using a different catalog? Replace `databricks_cpg` everywhere in both notebooks.
> Local-only? See [Local dev](#local-dev-optional) — only the dataset is
> reproducible locally; the agent needs Databricks.

---

## Background: the business problem

**Trade promotion is one of the largest line items on a CPG company's P&L** —
brands like Pepsi, P&G, Nestlé, and Unilever spend an estimated **15–25% of gross
revenue** on trade promotions: the temporary price cuts, in-store displays, and
retailer mailers/flyers that push their products at stores like Kroger, Walmart,
and Target. For a company Pepsi's size, that's *tens of billions of dollars a year.*

The painful part: **industry studies consistently find that a large share of that
spend — often cited around half — fails to pay for itself.** A promotion either
lifts sales enough to justify the discount and merchandising cost, or it doesn't.
The core questions a commercial/category manager has to answer every planning
cycle are deceptively simple:

- Which products and departments *actually* lift when we promote them?
- Which promotions are just **subsidizing sales we'd have made at full price**?
- Where should next quarter's limited trade-spend budget go to maximize ROI?

These are hard because the answer lives in **enormous, messy retail data** —
hundreds of millions of transaction line items joined against promotion-exposure
data (was there a display? a mailer?) and a product hierarchy. Answering them
traditionally means a data analyst writing SQL, building a dashboard, and a
manager waiting days for a one-off report.

### Why solving it matters

Even a **small** improvement in promo allocation is enormous in absolute dollars.
Shifting a few points of a multi-billion-dollar trade budget away from
promotions that don't lift and toward ones that do directly improves margin —
without selling a single extra unit. That's why **measuring trade-promotion
effectiveness ("promo lift" / "trade-spend ROI") is one of the highest-leverage
analytics problems in the entire CPG industry.**

### What makes this a CPG problem specifically

This is the exact shape of work a commercial-analytics team at **Pepsi or P&G**
does, modeled here on the public **Dunnhumby "The Complete Journey"** retail
dataset (real household-level transactions + promotion exposure from a grocery
retailer):

- **Promotion exposure data** (`causal` table): in-store `display` and `mailer`
  flags per product per week — exactly the merchandising levers a brand pays the
  retailer for.
- **Baseline vs. promoted sales**: the agent computes average sale value when a
  product was *not* on promo vs. *on* promo — the literal definition of "lift."
- **Department/category hierarchy** (`products` table): brands plan trade spend
  by category, so the analysis rolls up to department level.

### How Databricks is used to solve it

The whole point is turning that slow, analyst-gated workflow into a **governed,
self-serve, auditable AI agent** — and Databricks (Azure Databricks here)
provides every layer:

- **Delta Lake** stores the raw retail data as versioned, queryable tables — the
  scalable "lakehouse" foundation that can hold a real CPG company's
  hundreds-of-millions-of-rows transaction history.
- **Unity Catalog** governs access: the agent queries named
  `databricks_cpg.cpg_demo` tables it's permitted to see — not raw files. In a
  real deployment this is what keeps sensitive sales data compliant and audited
  while still letting an AI agent use it.
- **Spark SQL** runs the promo-lift / trend joins across the full dataset at
  scale, inside the lakehouse, instead of exporting data out to a spreadsheet.
- **Mosaic AI** (Databricks model serving) hosts the LLM
  (`databricks-meta-llama-3-3-70b-instruct`) that powers the agent — so the model
  runs *next to the governed data* rather than shipping data to an external API.
- **MLflow tracing** logs every agent run end-to-end: the reasoning, the tool
  calls, the SQL, the inputs and outputs. A manager (or auditor) can see *exactly*
  how the agent reached a recommendation — critical when that recommendation
  moves real budget.

Net effect: a category manager can ask in plain English *"where should we focus
trade spend?"*, get a numerically-grounded answer with a concrete next action in
seconds, and trust it because every step is governed and traceable.

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
