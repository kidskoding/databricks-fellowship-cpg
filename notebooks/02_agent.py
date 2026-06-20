# Databricks notebook source

# COMMAND ----------
# MAGIC %pip install databricks-langchain mlflow langchain langgraph

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import mlflow
import pandas as pd
from databricks_langchain import ChatDatabricks
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

mlflow.langchain.autolog()

# COMMAND ----------
# Tools that query governed Unity Catalog Delta tables
# Every call is traceable in MLflow — no black box

@tool
def get_promo_lift(department: str) -> str:
    """Get promotion sales lift vs baseline for a CPG department."""
    result = spark.sql(f"""
        SELECT
            p.DEPARTMENT,
            AVG(CASE WHEN COALESCE(c.display, '0') != '0' OR COALESCE(c.mailer, '0') != '0' THEN t.SALES_VALUE END) AS promo_sales,
            AVG(CASE WHEN COALESCE(c.display, '0')  = '0' AND COALESCE(c.mailer, '0')  = '0' THEN t.SALES_VALUE END) AS base_sales
        FROM databricks_cpg.cpg_demo.transactions t
        JOIN databricks_cpg.cpg_demo.products p
          ON t.PRODUCT_ID = p.PRODUCT_ID
        LEFT JOIN databricks_cpg.cpg_demo.causal c
          ON t.PRODUCT_ID = c.PRODUCT_ID AND t.WEEK_NO = c.WEEK_NO AND t.STORE_ID = c.STORE_ID
        WHERE UPPER(p.DEPARTMENT) = UPPER('{department}')
        GROUP BY p.DEPARTMENT
    """).toPandas()

    if result.empty:
        return f"No data for department: {department}"

    row = result.iloc[0]
    promo, base = row["promo_sales"], row["base_sales"]
    if pd.isna(base) or base == 0:
        return f"Department: {row['DEPARTMENT']} | No non-promo baseline sales — lift undefined."
    if pd.isna(promo):
        return f"Department: {row['DEPARTMENT']} | No on-promo sales — lift undefined."
    lift = (promo - base) / base * 100
    return (
        f"Department: {row['DEPARTMENT']} | "
        f"Base avg sale: ${base:.2f} | "
        f"Promo avg sale: ${promo:.2f} | "
        f"Lift: {lift:.1f}%"
    )

@tool
def get_top_departments() -> str:
    """List all CPG departments available in the dataset."""
    result = spark.sql("""
        SELECT DISTINCT DEPARTMENT
        FROM databricks_cpg.cpg_demo.products
        WHERE DEPARTMENT IS NOT NULL
        ORDER BY DEPARTMENT
    """).toPandas()
    return ", ".join(result["DEPARTMENT"].tolist())

@tool
def get_weekly_promo_trend(department: str) -> str:
    """Week-over-week total sales for a department, with the share of each week's sales that were on promotion."""
    result = spark.sql(f"""
        SELECT
            t.WEEK_NO,
            SUM(t.SALES_VALUE) AS total_sales,
            SUM(CASE WHEN COALESCE(c.display, '0') != '0' OR COALESCE(c.mailer, '0') != '0'
                     THEN t.SALES_VALUE ELSE 0 END) AS promo_sales
        FROM databricks_cpg.cpg_demo.transactions t
        JOIN databricks_cpg.cpg_demo.products p ON t.PRODUCT_ID = p.PRODUCT_ID
        LEFT JOIN databricks_cpg.cpg_demo.causal c ON t.PRODUCT_ID = c.PRODUCT_ID AND t.WEEK_NO = c.WEEK_NO AND t.STORE_ID = c.STORE_ID
        WHERE UPPER(p.DEPARTMENT) = UPPER('{department}')
        GROUP BY t.WEEK_NO
        ORDER BY t.WEEK_NO
    """).toPandas()

    if result.empty:
        return f"No trend data for: {department}"

    def fmt(r):
        share = (r["promo_sales"] / r["total_sales"] * 100) if r["total_sales"] else 0
        tag = f" (promo {share:.0f}%)" if share > 0 else ""
        return f"Wk{r['WEEK_NO']}{tag}: ${r['total_sales']:.0f}"

    return " | ".join(fmt(r) for r in result.to_dict("records"))

# COMMAND ----------
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")

tools = [get_top_departments, get_promo_lift, get_weekly_promo_trend]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a CPG commercial analytics agent for companies like Pepsi and P&G.
You have access to real retail transaction and promotion data governed by Unity Catalog.
Use tools to answer questions about promotion performance, trade spend effectiveness, and sales trends.
Always give specific numbers. Always recommend a concrete next action."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# COMMAND ----------
# Every run logged in MLflow — full reasoning trace, inputs, outputs, tool calls
mlflow.set_experiment("/cpg-promo-agent")

with mlflow.start_run(run_name="promo_lift_analysis"):
    response = agent_executor.invoke({
        "input": "Which CPG departments have the highest promotion lift? Where should Pepsi focus trade spend next quarter?"
    })
    print(response["output"])
