# Databricks notebook source

# COMMAND ----------
# MAGIC %pip install kagglehub

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import kagglehub
from pathlib import Path

path = kagglehub.dataset_download("frtgnn/dunnhumby-the-complete-journey")
print(f"Downloaded to: {path}")

# COMMAND ----------
spark.sql("USE CATALOG databricks_cpg")
spark.sql("CREATE SCHEMA IF NOT EXISTS cpg_demo")

# COMMAND ----------
transactions = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"file://{path}/transaction_data.csv"))

transactions.write.format("delta").mode("overwrite").saveAsTable("databricks_cpg.cpg_demo.transactions")
print(f"transactions: {transactions.count()} rows")

# COMMAND ----------
causal = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"file://{path}/causal_data.csv"))

causal.write.format("delta").mode("overwrite").saveAsTable("databricks_cpg.cpg_demo.causal")
print(f"causal: {causal.count()} rows")

# COMMAND ----------
products = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"file://{path}/product.csv"))

products.write.format("delta").mode("overwrite").saveAsTable("databricks_cpg.cpg_demo.products")
print(f"products: {products.count()} rows")

# COMMAND ----------
spark.sql("SHOW TABLES IN databricks_cpg.cpg_demo").show()
