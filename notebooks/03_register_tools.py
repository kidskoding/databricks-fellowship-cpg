# Databricks notebook source

# COMMAND ----------
# MAGIC %md
# MAGIC Registers the agent's three tools as **Unity Catalog SQL functions** in
# MAGIC `databricks_cpg.cpg_demo`. Unlike `spark.sql` in 02_agent.py, UC functions
# MAGIC execute server-side, so a Model Serving endpoint (which has no Spark
# MAGIC session) can still call them. Run this once on a cluster before deploying.
# MAGIC The function COMMENT and parameter COMMENTs become the tool descriptions
# MAGIC the LLM reads when choosing tools.

# COMMAND ----------
spark.sql("""
CREATE OR REPLACE FUNCTION databricks_cpg.cpg_demo.list_departments()
RETURNS STRING
COMMENT 'List all CPG departments available in the dataset, comma-separated and alphabetized.'
RETURN
  SELECT CONCAT_WS(', ', SORT_ARRAY(COLLECT_SET(DEPARTMENT)))
  FROM databricks_cpg.cpg_demo.products
  WHERE DEPARTMENT IS NOT NULL
""")

# COMMAND ----------
spark.sql("""
CREATE OR REPLACE FUNCTION databricks_cpg.cpg_demo.get_promo_lift(
  department STRING COMMENT 'CPG department name, e.g. GROCERY'
)
RETURNS STRING
COMMENT 'Promotion sales lift vs baseline for a CPG department: base avg sale, promo avg sale, and percent lift.'
RETURN
  COALESCE(
    (
      SELECT
        CASE
          WHEN agg.base_sales IS NULL OR agg.base_sales = 0
            THEN CONCAT('Department: ', agg.DEPARTMENT, ' | No non-promo baseline sales - lift undefined.')
          WHEN agg.promo_sales IS NULL
            THEN CONCAT('Department: ', agg.DEPARTMENT, ' | No on-promo sales - lift undefined.')
          ELSE CONCAT(
            'Department: ', agg.DEPARTMENT,
            ' | Base avg sale: $', FORMAT_NUMBER(agg.base_sales, 2),
            ' | Promo avg sale: $', FORMAT_NUMBER(agg.promo_sales, 2),
            ' | Lift: ', FORMAT_NUMBER((agg.promo_sales - agg.base_sales) / agg.base_sales * 100, 1), '%'
          )
        END
      FROM (
        SELECT
          p.DEPARTMENT,
          AVG(CASE WHEN COALESCE(c.display, '0') != '0' OR COALESCE(c.mailer, '0') != '0' THEN t.SALES_VALUE END) AS promo_sales,
          AVG(CASE WHEN COALESCE(c.display, '0')  = '0' AND COALESCE(c.mailer, '0')  = '0' THEN t.SALES_VALUE END) AS base_sales
        FROM databricks_cpg.cpg_demo.transactions t
        JOIN databricks_cpg.cpg_demo.products p
          ON t.PRODUCT_ID = p.PRODUCT_ID
        LEFT JOIN databricks_cpg.cpg_demo.causal c
          ON t.PRODUCT_ID = c.PRODUCT_ID AND t.WEEK_NO = c.WEEK_NO AND t.STORE_ID = c.STORE_ID
        WHERE UPPER(p.DEPARTMENT) = UPPER(get_promo_lift.department)
        GROUP BY p.DEPARTMENT
      ) AS agg
    ),
    CONCAT('No data for department: ', get_promo_lift.department)
  )
""")

# COMMAND ----------
spark.sql("""
CREATE OR REPLACE FUNCTION databricks_cpg.cpg_demo.get_weekly_promo_trend(
  department STRING COMMENT 'CPG department name, e.g. GROCERY'
)
RETURNS STRING
COMMENT 'Week-over-week total sales for a department, with the share of each week''s sales that were on promotion.'
RETURN
  COALESCE(
    NULLIF(
      (
        SELECT CONCAT_WS(' | ',
          TRANSFORM(
            ARRAY_SORT(
              COLLECT_LIST(STRUCT(
                wk.WEEK_NO AS week,
                CONCAT(
                  'Wk', CAST(wk.WEEK_NO AS STRING),
                  CASE WHEN wk.promo_share > 0 THEN CONCAT(' (promo ', FORMAT_NUMBER(wk.promo_share, 0), '%)') ELSE '' END,
                  ': $', FORMAT_NUMBER(wk.total_sales, 0)
                ) AS line
              )),
              (l, r) -> CASE WHEN l.week < r.week THEN -1 WHEN l.week > r.week THEN 1 ELSE 0 END
            ),
            x -> x.line
          )
        )
        FROM (
          SELECT
            t.WEEK_NO,
            SUM(t.SALES_VALUE) AS total_sales,
            CASE WHEN SUM(t.SALES_VALUE) > 0
                 THEN SUM(CASE WHEN COALESCE(c.display, '0') != '0' OR COALESCE(c.mailer, '0') != '0' THEN t.SALES_VALUE ELSE 0 END) / SUM(t.SALES_VALUE) * 100
                 ELSE 0 END AS promo_share
          FROM databricks_cpg.cpg_demo.transactions t
          JOIN databricks_cpg.cpg_demo.products p
            ON t.PRODUCT_ID = p.PRODUCT_ID
          LEFT JOIN databricks_cpg.cpg_demo.causal c
            ON t.PRODUCT_ID = c.PRODUCT_ID AND t.WEEK_NO = c.WEEK_NO AND t.STORE_ID = c.STORE_ID
          WHERE UPPER(p.DEPARTMENT) = UPPER(get_weekly_promo_trend.department)
          GROUP BY t.WEEK_NO
        ) AS wk
      ),
      ''
    ),
    CONCAT('No trend data for: ', get_weekly_promo_trend.department)
  )
""")

# COMMAND ----------
# Sanity-check the functions resolve and run
spark.sql("SELECT databricks_cpg.cpg_demo.list_departments() AS departments").show(truncate=False)
spark.sql("SELECT databricks_cpg.cpg_demo.get_promo_lift('GROCERY') AS lift").show(truncate=False)
