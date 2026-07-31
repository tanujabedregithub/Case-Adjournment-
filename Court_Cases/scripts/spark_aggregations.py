from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark import StorageLevel


# ─────────────────────────────────────────
# 1. SPARK SESSION
# WHY: We use jdbc to read from PostgreSQL
# so we need postgresql package
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("CourtCase-Aggregations") \
    .config("spark.jars.packages", "org.postgresql:postgresql:42.6.0") \
    .config("spark.network.timeout", "1200s") \
    .config("spark.executor.heartbeatInterval", "30s") \
    .config("spark.sql.shuffle.partitions", "64") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Aggregation Job Started")

# ─────────────────────────────────────────
# 2. POSTGRESQL CONNECTION
# WHY: All aggregations read from
# cases_cleaned and write back to
# their respective summary tables
# ─────────────────────────────────────────
PG_URL = "jdbc:postgresql://pipeline_postgres:5432/court_docket_db"
PG_PROPS = {
    "user"    : "court_user",
    "password": "court_password",
    "driver"  : "org.postgresql.Driver"
}

# ─────────────────────────────────────────
# 3. LOAD CASES_CLEANED
# WHY: This is our main table with all
# cleaned and enriched case data
# All aggregations are computed from this
# ─────────────────────────────────────────
print("Loading cases_cleaned...")

df = spark.read \
    .format("jdbc") \
    .option("url", PG_URL) \
    .option("dbtable", "cases_cleaned") \
    .option("user", "court_user") \
    .option("password", "court_password") \
    .option("driver", "org.postgresql.Driver") \
    .option("fetchsize", "10000") \
    .option("partitionColumn", "filing_year") \
    .option("lowerBound", 2010) \
    .option("upperBound", 2018) \
    .option("numPartitions", 8) \
    .load()

# df.persist(StorageLevel.MEMORY_AND_DISK)
total = df.count()
print(f"Total rows loaded: {total:,}")

# ─────────────────────────────────────────
# 4. YEARLY SUMMARY
# Shows trend of filing vs disposal
# year by year from 2010 to 2018
# Used for line chart in dashboard
# ─────────────────────────────────────────
print("Computing yearly summary...")

yearly = df.groupBy("filing_year").agg(
    f.count("*").alias("total_cases"),
    f.sum("is_resolved").alias("resolved_cases"),
    (f.count("*") - f.sum("is_resolved"))
        .alias("pending_cases"),
    f.round(f.avg("is_resolved") * 100, 2)
        .alias("resolution_rate"),
    f.round(f.avg("case_age_days"), 0)
        .alias("avg_case_age")
).orderBy("filing_year")

yearly.write.jdbc(
    url=PG_URL,
    table="yearly_summary",
    mode="overwrite",
    properties=PG_PROPS
)
print("yearly_summary saved")

# ─────────────────────────────────────────
# 5. STATE SUMMARY
# Shows which states have highest backlog
# Used for India map in dashboard
# ─────────────────────────────────────────
print("Computing state summary...")

state = df.groupBy("state_name").agg(
    f.count("*").alias("total_cases"),
    f.sum("is_resolved").alias("resolved_cases"),
    (f.count("*") - f.sum("is_resolved"))
        .alias("pending_cases"),
    f.round(f.avg("is_resolved") * 100, 2)
        .alias("resolution_rate"),
    f.round(f.avg("case_age_days"), 0)
        .alias("avg_case_age")
).orderBy(f.col("total_cases").desc())

state.write.jdbc(
    url=PG_URL,
    table="state_summary",
    mode="overwrite",
    properties=PG_PROPS
)
print("state_summary saved")

# ─────────────────────────────────────────
# 6. CASE TYPE SUMMARY
# Shows which case types take longest
# Used for bar chart in dashboard
# ─────────────────────────────────────────
print("Computing case type summary...")

casetype = df.groupBy("case_type").agg(
    f.count("*").alias("total_cases"),
    f.sum("is_resolved").alias("resolved_cases"),
    (f.count("*") - f.sum("is_resolved"))
        .alias("pending_cases"),
    f.round(f.avg("is_resolved") * 100, 2)
        .alias("resolution_rate"),
    f.round(
        f.avg(
            f.when(f.col("is_resolved") == 1,
                   f.col("case_age_days"))
        ), 0
    ).alias("avg_resolution_days")
).orderBy(f.col("total_cases").desc())

casetype.write.jdbc(
    url=PG_URL,
    table="casetype_summary",
    mode="overwrite",
    properties=PG_PROPS
)
print("casetype_summary saved")

# ─────────────────────────────────────────
# 7. DELAY CATEGORY SUMMARY
# Fast/Medium/Slow/Stuck distribution
# Used for donut chart in dashboard
# ─────────────────────────────────────────
print("Computing delay summary...")

delay = df.groupBy("delay_category").agg(
    f.count("*").alias("total_cases")
).withColumn(
    "percentage",
    f.round(f.col("total_cases") / total * 100, 2)
)

delay.write.jdbc(
    url=PG_URL,
    table="delay_summary",
    mode="overwrite",
    properties=PG_PROPS
)
print("delay_summary saved")

# ─────────────────────────────────────────
# 8. JUDGE SUMMARY
# Shows which judge positions clear
# most cases — unique insight in project
# ─────────────────────────────────────────
print("Computing judge summary...")

judge = df.groupBy("judge_position").agg(
    f.count("*").alias("total_cases"),
    f.sum("is_resolved").alias("resolved_cases"),
    f.round(f.avg("is_resolved") * 100, 2)
        .alias("resolution_rate"),
    f.round(
        f.avg(
            f.when(f.col("is_resolved") == 1,
                   f.col("case_age_days"))
        ), 0
    ).alias("avg_resolution_days")
).orderBy(f.col("total_cases").desc())

judge.write.jdbc(
    url=PG_URL,
    table="judge_summary",
    mode="overwrite",
    properties=PG_PROPS
)
print("judge_summary saved")

# ─────────────────────────────────────────
# 9. GENDER SUMMARY
# Shows resolution rate by petitioner gender
# Unique social insight — nobody else has this
# ─────────────────────────────────────────
print("Computing gender summary...")

gender = df.groupBy("petitioner_gender").agg(
    f.count("*").alias("total_cases"),
    f.sum("is_resolved").alias("resolved_cases"),
    f.round(f.avg("is_resolved") * 100, 2)
        .alias("resolution_rate"),
    f.round(
        f.avg(
            f.when(f.col("is_resolved") == 1,
                   f.col("case_age_days"))
        ), 0
    ).alias("avg_resolution_days")
)

gender.write.jdbc(
    url=PG_URL,
    table="gender_summary",
    mode="overwrite",
    properties=PG_PROPS
)
print("gender_summary saved")

# ─────────────────────────────────────────
# 10. TIME TRENDS
# Day/Week/Month/Year wise case counts
# Used for trend charts in dashboard
# ─────────────────────────────────────────
print("Computing time trends...")

# Year wise trend
year_trend = df.groupBy(
    f.lit("yearly").alias("period_type"),
    f.col("filing_year").cast(StringType())
        .alias("period_value")
).agg(
    f.count("*").alias("total_cases"),
    f.sum("is_resolved").alias("resolved_cases"),
    (f.count("*") - f.sum("is_resolved"))
        .alias("pending_cases"),
    f.round(f.avg("is_resolved") * 100, 2)
        .alias("resolution_rate")
)

# Month wise trend
month_trend = df.groupBy(
    f.lit("monthly").alias("period_type"),
    f.date_format(f.col("date_of_filing"), "yyyy-MM").alias("period_value")
).agg(
    f.count("*").alias("total_cases"),
    f.sum("is_resolved").alias("resolved_cases"),
    (f.count("*") - f.sum("is_resolved"))
        .alias("pending_cases"),
    f.round(f.avg("is_resolved") * 100, 2)
        .alias("resolution_rate")
)

# Combine both trends
time_trends = year_trend.union(month_trend)

time_trends.write.jdbc(
    url=PG_URL,
    table="time_trends",
    mode="overwrite",
    properties=PG_PROPS
)
print("time_trends saved")

# ─────────────────────────────────────────
# 11. DISTRICT SUMMARY
# Same as state summary but drilled down
# to district level — for map drill-down
# ─────────────────────────────────────────
print("Computing district summary...")

district = df.groupBy("state_name", "district_name").agg(
    f.count("*").alias("total_cases"),
    f.sum("is_resolved").alias("resolved_cases"),
    (f.count("*") - f.sum("is_resolved"))
        .alias("pending_cases"),
    f.round(f.avg("is_resolved") * 100, 2)
        .alias("resolution_rate"),
    f.round(f.avg("case_age_days"), 0)
        .alias("avg_case_age")
).orderBy(f.col("total_cases").desc())

district.write.jdbc(
    url=PG_URL,
    table="district_summary",
    mode="overwrite",
    properties=PG_PROPS
)
print("district_summary saved")

# ─────────────────────────────────────────
# 12. DISPOSAL TYPE SUMMARY
# Shows HOW cases end (dismissed, settled,
# etc.) — different insight from case_type
# ─────────────────────────────────────────
print("Computing disposal summary...")

disposal = df.groupBy("disposal_type").agg(
    f.count("*").alias("total_cases"),
    f.round(f.avg("case_age_days"), 0)
        .alias("avg_case_age"),
    f.round(
        f.avg(
            f.when(f.col("is_resolved") == 1,
                   f.col("case_age_days"))
        ), 0
    ).alias("avg_resolution_days")
).orderBy(f.col("total_cases").desc())

disposal.write.jdbc(
    url=PG_URL,
    table="disposal_summary",
    mode="overwrite",
    properties=PG_PROPS
)
print("disposal_summary saved")

# ─────────────────────────────────────────
# 13. HEARING SPAN SUMMARY
# Avg gap between first and last hearing,
# by case type and delay category — shows
# whether "Stuck" cases are stuck due to
# hearing gaps or just old filing dates
# ─────────────────────────────────────────
print("Computing hearing span summary...")

hearing_span = df.filter(
    f.col("hearing_span_days").isNotNull()
).groupBy("case_type", "delay_category").agg(
    f.count("*").alias("total_cases"),
    f.round(f.avg("hearing_span_days"), 0)
        .alias("avg_hearing_span_days"),
    f.round(f.max("hearing_span_days"), 0)
        .alias("max_hearing_span_days")
).orderBy(f.col("total_cases").desc())

hearing_span.write.jdbc(
    url=PG_URL,
    table="hearing_span_summary",
    mode="overwrite",
    properties=PG_PROPS
)
print("hearing_span_summary saved")

# ─────────────────────────────────────────
# 14. FEMALE DEFENDANT SUMMARY
# Mirror of gender_summary but for
# defendant side instead of petitioner
# ─────────────────────────────────────────
print("Computing defendant gender summary...")

defendant_gender_summary = df.groupBy("defendant_gender").agg(
    f.count("*").alias("total_cases"),
    f.sum("is_resolved").alias("resolved_cases"),
    f.round(f.avg("is_resolved") * 100, 2).alias("resolution_rate"),
    f.round(
        f.avg(
            f.when(f.col("is_resolved") == 1, f.col("case_age_days"))
        ),
        0
    ).alias("avg_resolution_days")
)

defendant_gender_summary.write.jdbc(
    url=PG_URL,
    table="defendant_gender_summary",
    mode="overwrite",
    properties=PG_PROPS
)
print("defendant_gender_summary saved")

# ─────────────────────────────────────────
# 15. TOP OLDEST PENDING CASES
# Drill-through table — the longest-running
# unresolved cases across the dataset
# ─────────────────────────────────────────
print("Computing oldest pending cases...")

oldest_pending = df.filter(
    f.col("is_resolved") == 0
).select(
    "ddl_case_id",
    "state_name",
    "district_name",
    "court_name",
    "case_type",
    "date_of_filing",
    "case_age_days",
    "delay_category"
).orderBy(f.col("case_age_days").desc()).limit(50)

oldest_pending.write.jdbc(
    url=PG_URL,
    table="oldest_pending_cases",
    mode="overwrite",
    properties=PG_PROPS
)
print("oldest_pending_cases saved")

# ─────────────────────────────────────────
# 16. CASE TYPE x DELAY CATEGORY CROSS-TAB
# Heatmap-ready: proportion of each case
# type falling into Fast/Medium/Slow/Stuck
# ─────────────────────────────────────────
print("Computing case type x delay crosstab...")

crosstab = df.groupBy("case_type").pivot(
    "delay_category", ["Fast", "Medium", "Slow", "Stuck"]
).agg(f.count("*")).fillna(0)

# add a total column for computing percentages
# in Power BI without extra joins
crosstab = crosstab.withColumn(
    "total_cases",
    f.col("Fast") + f.col("Medium") + f.col("Slow") + f.col("Stuck")
).orderBy(f.col("total_cases").desc())

crosstab.write.jdbc(
    url=PG_URL,
    table="casetype_delay_crosstab",
    mode="overwrite",
    properties=PG_PROPS
)
print("casetype_delay_crosstab saved")



# ─────────────────────────────────────────
# 11. FINAL SUMMARY
# ─────────────────────────────────────────
print("\n========== AGGREGATION COMPLETE ==========")
print(f"Total cases processed : {total:,}")
resolved = df.filter(f.col("is_resolved") == 1).count()
print(f"Resolved cases : {resolved:,}")
print(f"Pending cases  : {total - resolved:,}")
print("All summary tables saved to PostgreSQL")
print("==========================================")

# df.unpersist()
spark.stop()