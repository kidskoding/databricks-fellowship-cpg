# CPG Trade Promotion Agent — Project Notes

A plain-language guide to what this project is, how it works, and everything that
has been changed so far.

---

## 1. What the project is

A conversational analytics agent for the CPG (Consumer Packaged Goods) industry —
companies like Pepsi and P&G. A manager asks a plain-English question
("which departments lift most when promoted, and where should we spend next
quarter?") and the agent answers with real numbers from real retail data, then
recommends an action.

**The business problem:** CPG brands spend 15–25% of revenue paying stores to
discount and display products ("trade promotion"), and roughly half of that spend
is wasted on shoppers who would have bought anyway. The agent measures **promo
lift** — average sales on promotion vs. baseline — to find which promotions are
worth funding.

It runs entirely on Databricks. That is the real point: it is a reference pattern
for a governed, auditable AI agent that lives next to the company's data.

---

## 2. Architecture

```
Kaggle (Dunnhumby grocery dataset)
        │   01_setup.py downloads it
        ▼
3 Delta tables: transactions · causal · products      (stored in the lakehouse)
        │                                              governed by Unity Catalog
        ▼
3 tools = 3 SQL queries the agent is allowed to run
        │
        ▼
LLM (Llama 3.3 70B, served by Databricks) picks tools, reads results
        │
        ▼
A grounded answer + recommendation     every step logged in MLflow (audit trail)
```

| Layer | Role | Why it exists |
| --- | --- | --- |
| Delta Lake | Versioned, queryable tables | Cheap file storage plus database reliability |
| Unity Catalog | Governance by naming | The agent touches named tables it is permitted to, not raw files |
| Spark SQL | Query engine | Joins and aggregates the data |
| Mosaic AI | Hosts the LLM | The model runs next to the data, not on an external API |
| MLflow | Tracing | Records every step so a recommendation is auditable, not a black box |

**The three tables only make sense joined together:** `transactions` has the money
(what sold, when, for how much), `causal` has the promotions (was there a display
or mailer that week, per store), `products` has the categories (which department a
product is in). The SQL stitches them on shared keys to ask "did the promotion
actually move sales?"

**Tool-calling is the key idea.** An LLM on its own would invent (hallucinate)
numbers. Instead it is given tools (functions that run real SQL), it decides which
to call, reads the true results, and only then writes the answer. Its job shrinks
to choosing what to look up and narrating verified results.

---

## 3. The files

| File | What it does |
| --- | --- |
| `notebooks/01_setup.py` | Downloads the dataset → writes 3 Delta tables in Unity Catalog |
| `notebooks/02_agent.py` | The interactive agent: LangChain tool-calling + MLflow tracing |
| `notebooks/03_register_tools.py` | The 3 tools as governed Unity Catalog SQL functions (for serving) |
| `notebooks/agent.py` | The servable MLflow ChatAgent (calls the UC functions) |
| `notebooks/04_deploy.py` | Logs → registers to UC → deploys (serving endpoint + chat UI) |
| `databricks.yml`, `resources/cpg_pipeline.job.yml` | Bundle: one job running setup → register_tools → deploy_agent |
| `.github/workflows/deploy.yml` | CI/CD: bundle validate → deploy → run on push to master |

---

## 4. What has been done

### Step 1 — Understood it deeply
Read the whole repo, broke it into concept layers, and fact-checked each against
the actual code. That surfaced real bugs hiding in clean-looking SQL.

### Step 2 — The "checker" question (Phoenix vs MLflow)
Considered using Arize Phoenix as a checker. Conclusion: Phoenix is a better
answer-quality checker than MLflow's passive tracing, but neither checks the
correctness of the SQL math — so it would not catch this project's real problems.
Decision: do not bolt Phoenix on; fix the SQL instead.

### Step 3 — Fixed 6 bugs (PR #1, merged to master)
One commit each:

| # | Bug | Why it mattered |
| --- | --- | --- |
| 1 | Join ignored `STORE_ID` | `causal` is per-store; joining on only product+week duplicated every sale across stores, distorting the numbers |
| 2 | `LEFT JOIN` NULLs dropped from both buckets | Sales with no promo record vanished instead of counting as baseline |
| 3 | `Lift: nan%` | A `NaN` slipped past the guard; printed `nan` instead of a clean "undefined" |
| 4 | Weekly trend `LIMIT 10` + loose flag | Returned only the first 10 weeks and flagged almost every week as "promo" |
| 5 | SQL injection | The department string was pasted straight into the query |
| 6 | `get_top_departments` misnomer | It ranks nothing — renamed to `list_departments` |

### Step 4 — Built serving + chat UI (PR #2, merged)
Goal: a UI. Chose to deploy natively on Databricks Model Serving, which provides a
built-in chat interface (AI Playground / Review App).

Key constraint: a serving endpoint has no Spark session, so the original
`spark.sql(...)` tools cannot run when served. Fixed the most-governed way — the
three tools became Unity Catalog SQL functions that execute server-side.

How the served version works end to end:
1. Run `01_setup.py` (data) and `03_register_tools.py` (creates the SQL functions).
2. `04_deploy.py` packages `agent.py`, registers it in Unity Catalog, and deploys it.
3. The deploy creates a serving endpoint (REST API) and a Review App (chat box).
4. A user asks a question → the LLM calls the UC functions → governed SQL runs →
   the LLM writes a grounded answer → MLflow records the whole trace.

---

### Step 5 — First real run on Databricks + CI/CD
Ran the whole pipeline on the workspace (serverless only, no cluster). The first
real run surfaced five breaks, each fixed:

| # | Break | Fix |
| --- | --- | --- |
| 1 | Serverless can't read driver-local `file://` paths | `01_setup.py` stages CSVs in UC volume `databricks_cpg.cpg_demo.raw` |
| 2 | LangChain 1.x removed `AgentExecutor` | Both agents use `langchain.agents.create_agent` |
| 3 | Unpinned `%pip install -U` hung 20+ min in pip backtracking | Exact version pins in every notebook + `log_model` |
| 4 | UC toolkit marks `department` optional; Llama sometimes omits it and the exception killed the request | Middleware hands tool errors back to the LLM so it retries |
| 5 | `agents.deploy(scale_to_zero_enabled=True)` silently ignored | Correct kwarg is `scale_to_zero=True` |

Also: blank `" "` department filtered out of `list_departments`; catalog is
created via SQL (the REST API needs a storage location on Default Storage
workspaces).

CI/CD added: a Databricks Asset Bundle (`databricks.yml` +
`resources/cpg_pipeline.job.yml`) defines the job `setup → register_tools →
deploy_agent`; a GitHub Actions workflow validates, deploys and re-runs
`register_tools,deploy_agent` on every push to `master`.

---

## 5. Current state and honest caveats

- **Running:** tables loaded (2.6M transactions, 36.8M causal rows, 92K
  products), UC functions live, agent registered in UC and served on endpoint
  `agents_databricks_cpg-cpg_demo-cpg_promo_agent` with a Review App chat UI.
- **CI needs two GitHub secrets** (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`) before
  the workflow can deploy.
- **Real numbers differ from the old README sample:** GROCERY lift is -6.1%,
  SPIRITS +36.7%. The old sample (+27.1% GROCERY) was illustrative, not real.
- **Agent reasoning is shallow:** it may call a department "highest" after
  checking two. The numbers are real; the ranking claim is the LLM's.
- **Known limitation left in place:** "lift" is still average sale value per line,
  not true incremental volume. Fixing it is a redesign (needs quantity + a causal
  model), not a bug fix.

**The throughline:** took a slick-looking but quietly broken notebook demo, made
the numbers correct, then gave it a real front door — without ever leaving the
governed Databricks platform the project is meant to showcase.
