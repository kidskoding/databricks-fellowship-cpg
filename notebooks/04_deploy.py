# Databricks notebook source

# COMMAND ----------
# MAGIC %md
# MAGIC Driver notebook: logs `agent.py` as an MLflow model, registers it to Unity
# MAGIC Catalog, and deploys it with `agents.deploy(...)`. That one deploy call
# MAGIC creates the serving endpoint **and** the Review App / AI Playground chat UI,
# MAGIC with MLflow tracing on. Run `03_register_tools.py` first.

# COMMAND ----------
# MAGIC %pip install -U "mlflow>=3.1.3" "databricks-agents>=1.1.0" databricks-langchain "unitycatalog-ai[databricks]" langchain

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import mlflow
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksServingEndpoint,
    DatabricksTable,
)

CATALOG, SCHEMA = "databricks_cpg", "cpg_demo"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
UC_MODEL = f"{CATALOG}.{SCHEMA}.cpg_promo_agent"

# Declares the resources the served agent needs, so Databricks can mint
# scoped, short-lived credentials for them (automatic auth passthrough).
resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
    DatabricksFunction(function_name=f"{CATALOG}.{SCHEMA}.list_departments"),
    DatabricksFunction(function_name=f"{CATALOG}.{SCHEMA}.get_promo_lift"),
    DatabricksFunction(function_name=f"{CATALOG}.{SCHEMA}.get_weekly_promo_trend"),
    DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.transactions"),
    DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.causal"),
    DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.products"),
]

input_example = {
    "messages": [
        {"role": "user", "content": "Which CPG departments have the highest promotion lift? Where should Pepsi focus trade spend next quarter?"}
    ]
}

# COMMAND ----------
# Real-time tracing needs a non-Git experiment; set one explicitly.
mlflow.set_experiment("/Users/" + spark.sql("SELECT current_user()").collect()[0][0] + "/cpg-promo-agent")

with mlflow.start_run():
    logged = mlflow.pyfunc.log_model(
        name="agent",                 # MLflow 3; use artifact_path="agent" on MLflow 2.x
        python_model="agent.py",      # path is relative to this driver notebook
        input_example=input_example,
        resources=resources,
        pip_requirements=["mlflow>=3.1.3", "databricks-langchain", "unitycatalog-ai[databricks]", "langchain"],
    )

print("Model URI:", logged.model_uri)

# COMMAND ----------
mlflow.set_registry_uri("databricks-uc")
uc_info = mlflow.register_model(model_uri=logged.model_uri, name=UC_MODEL)
print("Registered:", UC_MODEL, "version", uc_info.version)

# COMMAND ----------
from databricks import agents

deployment = agents.deploy(UC_MODEL, uc_info.version, scale_to_zero_enabled=True)
print("Query endpoint:", deployment.query_endpoint)
# The Review App / AI Playground chat UI link is shown on the endpoint's page
# in Serving, and via: from databricks.agents import list_deployments; list_deployments()
