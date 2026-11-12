# Databricks notebook source

# COMMAND ----------
# MAGIC %pip install kagglehub

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import os
import shutil

CATALOG, SCHEMA = "databricks_cpg", "cpg_demo"
RAW = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
FILES = ["transaction_data.csv", "causal_data.csv", "product.csv"]

# Catalogs on Default Storage workspaces must be created via SQL (the REST API needs a location).
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw")

# COMMAND ----------
# Stage the CSVs in a UC volume: serverless compute can't read driver-local file:// paths.
# If this workspace has no internet egress, upload the 3 CSVs to the volume yourself
# (e.g. `databricks fs cp transaction_data.csv dbfs:/Volumes/databricks_cpg/cpg_demo/raw/`).
missing = [f for f in FILES if not os.path.exists(f"{RAW}/{f}")]
if missing:
    import kagglehub
    path = kagglehub.dataset_download("frtgnn/dunnhumby-the-complete-journey")
    for f in missing:
        shutil.copy(f"{path}/{f}", f"{RAW}/{f}")
print(os.listdir(RAW))

# COMMAND ----------
def load(csv, table):
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{RAW}/{csv}")
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.{table}")
    print(f"{table}: {spark.table(f'{CATALOG}.{SCHEMA}.{table}').count()} rows")

load("transaction_data.csv", "transactions")
load("causal_data.csv", "causal")
load("product.csv", "products")

# COMMAND ----------
spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").show()
