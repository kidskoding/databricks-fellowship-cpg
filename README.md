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

> Runs **inside a Databricks workspace** (needs Unity Catalog + the
> `databricks-meta-llama-3-3-70b-instruct` serving endpoint). It is not a local app.
> Tested on serverless compute.

**One command (Declarative Automation Bundle):**

```bash
databricks bundle deploy -t prod
databricks bundle run cpg_pipeline -t prod   # load data → register tools → deploy agent + chat UI
```

Edit `workspace.host` in `databricks.yml` to point at your workspace.

**Or by hand, notebook by notebook:**

1. **Setup** — run `notebooks/01_setup.py`. It creates `databricks_cpg.cpg_demo`,
   stages the Dunnhumby CSVs in the volume `databricks_cpg.cpg_demo.raw`
   (downloading from Kaggle if they aren't there yet), and writes three Delta
   tables: `transactions`, `causal`, `products`.
2. **Run the agent** — run `notebooks/02_agent.py`. The last cell asks a sample
   question and prints the answer; the full trace lands in the MLflow experiment
   `/Users/<you>/cpg-promo-agent`.
3. **Serve it** — run `03_register_tools.py` then `04_deploy.py` (see
   [Serve it with a UI](#serve-it-with-a-ui-model-serving--review-app)).

> Using a different catalog? Replace `databricks_cpg` everywhere in the notebooks.
> No internet egress on your compute? Upload the 3 CSVs to the volume yourself:
> `databricks fs cp transaction_data.csv dbfs:/Volumes/databricks_cpg/cpg_demo/raw/`.

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

The core is two Databricks notebooks, run in order. Notebooks 03/04 serve the agent (see below).

1. **`notebooks/01_setup.py`** — Data ingestion
   - Stages the [Dunnhumby "The Complete Journey"](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey)
     retail dataset in a Unity Catalog volume (downloads it via `kagglehub` if missing).
   - Writes three CSVs into **Delta tables** under the Unity Catalog schema
     `databricks_cpg.cpg_demo`:
     - `transactions` — line-item sales (household, product, week, sales value)
     - `causal` — promotion exposure (in-store `display`, `mailer` flags)
     - `products` — product → department/commodity hierarchy

2. **`notebooks/02_agent.py`** — The agent
   - Builds a **tool-calling LangChain agent** (`langchain.agents.create_agent`) powered by a Databricks-hosted LLM
     (`databricks-meta-llama-3-3-70b-instruct` via `ChatDatabricks`).
   - Exposes three tools, each a governed Spark SQL query over the Delta tables:
     - `list_departments()` — list available CPG departments
     - `get_promo_lift(department)` — promo avg sale vs. baseline avg sale + % lift
     - `get_weekly_promo_trend(department)` — week-over-week sales with each week's promo share
   - Wraps everything in **MLflow autolog** (`mlflow.langchain.autolog()`), so each
     run captures the full reasoning trace, tool calls, inputs, and outputs under
     the experiment `/Users/<you>/cpg-promo-agent`. No black box.

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
      │  01_setup.py (kagglehub → UC volume)
      ▼
Delta tables  ──  databricks_cpg.cpg_demo.{transactions, causal, products}
      │              (governed by Unity Catalog)
      ▼
@tool functions (Spark SQL)  ──  list_departments / get_promo_lift / get_weekly_promo_trend
      │
      ▼
LangChain create_agent  +  ChatDatabricks (Llama 3.3 70B)
      │
      ▼
MLflow trace  ──  experiment /Users/<you>/cpg-promo-agent  (full reasoning + tool calls logged)
```

---

## Sample agent output

A real run of `02_agent.py` (serverless, Llama 3.3 70B, Nov 2026). The trace below
is read back from MLflow: the agent picks tools on its own, recovers from a bad
guess, and grounds its answer in governed SQL.

**Input**

```
Which CPG departments have the highest promotion lift?
Where should Pepsi focus trade spend next quarter?
```

**Tool calls** (from the MLflow trace, 28 s end to end)

```
get_promo_lift({'department': 'Beverages'})  -> No data for department: Beverages
list_departments()                            -> AUTOMOTIVE, CHARITABLE CONT, CHEF SHOPPE, ... GROCERY, ... SPIRITS, ...
get_promo_lift({'department': 'GROCERY'})     -> Department: GROCERY | Base avg sale: $2.52 | Promo avg sale: $2.37 | Lift: -6.1%
get_promo_lift({'department': 'SPIRITS'})     -> Department: SPIRITS | Base avg sale: $10.10 | Promo avg sale: $13.80 | Lift: 36.7%
```

**Final answer**

```
Based on the promotion lift data, Pepsi should focus trade spend on the SPIRITS
department next quarter, as it has the highest promotion lift of 36.7%. This
indicates that promotional activities in the SPIRITS department are effective
in increasing sales.
```

Note the agent compared only two departments before calling SPIRITS "the
highest" — the numbers are real, but the ranking claim is the LLM's. The trace
makes that visible, which is the point of tracing.

---

## Running it

These notebooks run **inside a Databricks workspace**, not locally. They rely on
Databricks runtime globals (`spark`, `dbutils`) and Databricks model serving
endpoints. They work on serverless compute (no cluster needed).

### Prerequisites
- A Databricks workspace with **Unity Catalog** enabled.
- Permission to create the catalog `databricks_cpg` (or edit the notebooks to use yours).
- Access to the `databricks-meta-llama-3-3-70b-instruct` serving endpoint.

### Package versions
Notebooks pin exact versions (`mlflow==3.18.0`, `langchain==1.5.0`,
`langgraph==1.2.21`, `databricks-langchain==0.22.0`, `databricks-agents==1.13.0`,
`unitycatalog-ai==0.4.1`). Unpinned `%pip install -U` sent pip into resolver
backtracking for 20+ minutes on serverless; LangChain 1.x also removed
`AgentExecutor`, which the agent no longer uses.

---

## Serve it with a UI (Model Serving + Review App)

`02_agent.py` runs the agent interactively in a notebook. To get a chat UI,
deploy it as a Mosaic AI agent — the deploy step creates a serving endpoint
**and** a Review App / AI Playground chat interface, with MLflow tracing on.

Serving endpoints have no Spark session, so the tools are exposed as
**Unity Catalog SQL functions** instead of `spark.sql`:

1. Run `notebooks/03_register_tools.py` to create the three UC functions
   (`list_departments`, `get_promo_lift`, `get_weekly_promo_trend`) in
   `databricks_cpg.cpg_demo`.
2. `notebooks/agent.py` is the servable agent (an MLflow `ChatAgent` that
   calls those UC functions via the UC function toolkit).
3. Run `notebooks/04_deploy.py` to log the agent, register it to Unity
   Catalog, and `agents.deploy(...)` it. The printed query endpoint and the
   Review App (linked on the endpoint's page) are your UI.

## CI/CD (GitHub Actions + bundle)

`databricks.yml` + `resources/cpg_pipeline.job.yml` define one job,
`cpg-promo-agent-pipeline`: `setup → register_tools → deploy_agent`, all on
serverless.

`.github/workflows/deploy.yml` runs on every push to `master` (or by hand):

1. `databricks bundle validate --strict -t prod`
2. `databricks bundle deploy -t prod` — syncs the notebooks + job definition
3. `databricks bundle run cpg_pipeline -t prod --only register_tools,deploy_agent`
   — re-creates the UC functions and redeploys the agent. The slow data load
   (`setup`) is skipped; run the full job by hand when the data changes.

Add two repo secrets: `DATABRICKS_HOST` (workspace URL) and `DATABRICKS_TOKEN`
(a personal access token, or better, a service principal's token).

## Repo layout

```
.
├── notebooks/
│   ├── 01_setup.py            # download dataset → Delta tables in Unity Catalog
│   ├── 02_agent.py            # LangChain tool-calling agent + MLflow tracing (interactive)
│   ├── 03_register_tools.py   # create the 3 tools as governed Unity Catalog SQL functions
│   ├── agent.py               # servable MLflow ChatAgent (calls the UC functions)
│   └── 04_deploy.py           # log → register to UC → deploy (serving endpoint + Review App)
├── resources/
│   └── cpg_pipeline.job.yml   # bundle job: setup → register_tools → deploy_agent
├── .github/workflows/
│   └── deploy.yml             # CI/CD: validate → deploy → run on push to master
├── databricks.yml             # bundle config (target: prod)
├── pyproject.toml             # local dev deps (uv) — mainly kagglehub
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

- **SQL injection:** fixed — `02_agent.py` uses parameterized `spark.sql(..., args=...)`
  and the UC functions take `department` as a typed parameter.
- **Catalog name** `databricks_cpg` is hardcoded — change it everywhere if your
  workspace differs.
- Promotion detection is heuristic: a row counts as "on promo" when `display` or
  `mailer` is non-`'0'` in the `causal` table.
