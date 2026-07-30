import os
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql.functions import broadcast
from pyspark.sql.types import *
from pyspark import StorageLevel

# ─────────────────────────────────────────
# 1. START SPARK SESSION
# 
# WHY these configs:
# - kafka package: allows Spark to read from Kafka
# - postgresql package: allows Spark to write to PostgreSQL
# - legacy timeParser: handles date formats in our data
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("CourtCase-Streaming") \
    .config("spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
        "org.postgresql:postgresql:42.6.0"
    ) \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()
    
spark.sparkContext.setLogLevel("WARN")
print("Spark Streaming Session Started")

# ─────────────────────────────────────────
# 2. LOAD KEY FILES USING BROADCAST
#
# WHY broadcast:
# Key files are small (few KB).
# Broadcasting sends one copy to every
# Spark worker so joins happen in memory
# without shuffling 40 million rows.
# This is much faster than normal join.
# ─────────────────────────────────────────
BASE = "/opt/spark/work-dir/data/csv"
print("Loading key files...")

type_key = spark.read.csv(
    f"{BASE}/keys/type_name_key.csv",
    header=True, inferSchema=True
).select(
    f.col("type_name").cast(IntegerType()),
    "type_name_s"
).dropDuplicates(["type_name"])

disp_key = spark.read.csv(
    f"{BASE}/keys/disp_name_key.csv",
    header=True, inferSchema=True
).select(
    f.col("disp_name").cast(IntegerType()),
    "disp_name_s"
).dropDuplicates(["disp_name"])

purpose_key = spark.read.csv(
    f"{BASE}/keys/purpose_name_key.csv",
    header=True, inferSchema=True
).select(
    f.col("purpose_name").cast(IntegerType()),
    "purpose_name_s"
).dropDuplicates(["purpose_name"])

state_key = spark.read.csv(
    f"{BASE}/keys/cases_state_key.csv",
    header=True, inferSchema=True
).select(
    f.col("state_code").cast(IntegerType()),
    "state_name"
).dropDuplicates(["state_code"])

district_key = spark.read.csv(
    f"{BASE}/keys/cases_district_key.csv",
    header=True, inferSchema=True
).select(
    f.col("state_code").cast(IntegerType()),
    f.col("dist_code").cast(IntegerType()),
    "district_name"
).dropDuplicates(["state_code", "dist_code"])

# WHY Window for court key:
# Same court code had different names in different years
# (example: J&K split into 2 states in 2019)
# We keep only the LATEST year's name using Window

court_key_raw = spark.read.csv(
    f"{BASE}/keys/cases_court_key.csv",
    header=True,
    inferSchema=True
)

window_spec = Window \
    .partitionBy("state_code", "dist_code", "court_no") \
    .orderBy(f.col("year").desc())

court_key = court_key_raw \
    .withColumn("row_num",
                f.row_number().over(window_spec)) \
    .filter(f.col("row_num") == 1) \
    .select(
        f.col("state_code").cast(IntegerType()),
        f.col("dist_code").cast(IntegerType()),
        f.col("court_no").cast(IntegerType()),
        "court_name"
    ) \
    .dropDuplicates(["state_code", "dist_code", "court_no"])

print("Key files loaded successfully")

# ─────────────────────────────────────────
# 3. DEFINE SCHEMA FOR KAFKA MESSAGES
#
# WHY explicit schema:
# Kafka sends data as raw text (JSON string).
# We tell Spark exactly what columns and types
# to expect so it doesn't guess wrong.
#
# WHY numeric-code fields are StringType here:
# csv.DictReader in the producer sends every field
# as a string, so JSON serializes them as quoted
# strings (e.g. "01" not 01). If this schema declared
# them as IntegerType, from_json would silently null
# them out on a type mismatch -- which is exactly
# what caused every join to fail earlier. We parse
# everything as string first, then cast explicitly
# right after (see below).
# ─────────────────────────────────────────
schema = StructType([
    StructField("ddl_case_id",       StringType(),  True),
    StructField("state_code",        StringType(),  True),
    StructField("dist_code",         StringType(),  True),
    StructField("court_no",          StringType(),  True),
    StructField("judge_position",    StringType(),  True),
    StructField("type_name",         StringType(),  True),
    StructField("purpose_name",      StringType(),  True),
    StructField("disp_name",         StringType(),  True),
    StructField("date_of_filing",    StringType(),  True),
    StructField("date_of_decision",  StringType(),  True),
    StructField("date_first_list",   StringType(),  True),
    StructField("date_last_list",    StringType(),  True),
    StructField("date_next_list",    StringType(),  True),
    StructField("female_petitioner", StringType(),  True),
    StructField("female_defendant",  StringType(),  True)
])

# ─────────────────────────────────────────
# 4. READ FROM KAFKA
#
# WHY these options:
# - pipeline_kafka:29092 is internal Docker network address
#   (containers talk to each other using container names)
# - startingOffsets earliest: read ALL messages
#   from the beginning, not just new ones
# - maxOffsetsPerTrigger: process 100,000 messages
#   per batch so Spark is not overwhelmed
# ─────────────────────────────────────────
print("Connecting to Kafka...")

kafka_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "court_cases_raw") \
    .option("startingOffsets", "earliest") \
    .option("maxOffsetsPerTrigger", 100000) \
    .load()

# WHY: Kafka sends each message as binary (bytes)
# We cast it to string first, then parse as JSON
# Debug stream (optional, disabled -- re-reads the
# entire topic from earliest with no checkpoint if
# ever re-enabled, so leave commented out)
# debug_query = kafka_stream.selectExpr("CAST(value AS STRING)") \
#     .writeStream \
#     .format("console") \
#     .option("truncate", False) \
#     .start()

# Parse JSON from Kafka
parsed = kafka_stream.select(
    f.from_json(
        f.col("value").cast("string"),
        schema
    ).alias("data")
).select("data.*")

# Cast numeric-code fields now that they're safely parsed as strings first
parsed = parsed \
    .withColumn("state_code",   f.col("state_code").cast(IntegerType())) \
    .withColumn("dist_code",    f.col("dist_code").cast(IntegerType())) \
    .withColumn("court_no",     f.col("court_no").cast(IntegerType())) \
    .withColumn("type_name",    f.col("type_name").cast(IntegerType())) \
    .withColumn("purpose_name", f.col("purpose_name").cast(IntegerType())) \
    .withColumn("disp_name",    f.col("disp_name").cast(IntegerType()))

print("Kafka connection established")

# ─────────────────────────────────────────
# 5. PROCESS EACH BATCH
#
# WHY foreachBatch:
# Spark Streaming processes data in small batches
# This function runs automatically for EACH batch
# epoch_id is the batch number (0, 1, 2, 3...)
# ─────────────────────────────────────────
def process_batch(df, epoch_id):
    print(f"\nProcessing batch {epoch_id}...")

    # WHY isEmpty() instead of count() == 0:
    # isEmpty() short-circuits after finding one row.
    # count() scans the whole batch just to check
    # if it's empty -- wasteful for every single batch.
    if df.isEmpty():
        print(f"Batch {epoch_id}: empty, skipping")
        return

    # ── STEP A: BROADCAST JOINS ──
    # Decode all number codes to readable text
    df = df.join(broadcast(type_key),
                 on="type_name", how="left")
    df = df.join(broadcast(disp_key),
                 on="disp_name", how="left")
    df = df.join(broadcast(purpose_key),
                 on="purpose_name", how="left")
    df = df.join(broadcast(state_key),
                 on="state_code", how="left")
    df = df.join(broadcast(district_key),
                 on=["state_code", "dist_code"],
                 how="left")
    df = df.join(broadcast(court_key),
                 on=["state_code", "dist_code", "court_no"],
                 how="left")

    # ── STEP B: CONVERT DATE COLUMNS ──
    # WHY: Dates arrive as text strings from Kafka
    # We convert to proper DATE type for calculations
    date_cols = [
        "date_of_filing", "date_of_decision",
        "date_first_list", "date_last_list",
        "date_next_list"
    ]
    for c in date_cols:
        df = df.withColumn(
            c,
            f.when(
                f.col(c).rlike(r"^\d{4}-\d{2}-\d{2}$"),
                f.to_date(f.col(c), "yyyy-MM-dd")
            ).otherwise(None)
        )

    # ── STEP C: CLEAN GENDER COLUMNS ──
    # WHY: Raw values are "1 female", "0 male", "-9998 unclear"
    # We simplify to just: female, male, unknown
    gender_cols = ["female_petitioner", "female_defendant"]
    for c in gender_cols:
        df = df.withColumn(c,
            f.when(f.col(c).cast("string")
                    .startswith("1"), "female")
            .when(f.col(c).cast("string")
                    .startswith("0"), "male")
            .otherwise("unknown")
        )

    # ── STEP D: FILL NULL VALUES ──
    # WHY: Some codes don't match key files
    # We fill with meaningful default values
    # instead of leaving them as NULL
    df = df.fillna({
        "purpose_name_s" : "Not Listed",
        "type_name_s"    : "Unknown",
        "disp_name_s"    : "Pending",
        "state_name"     : "Unknown",
        "district_name"  : "Unknown",
        "court_name"     : "Unknown"
    })

    # ── STEP E: REMOVE IMPOSSIBLE RECORDS ──
    # WHY: Some records have decision_date BEFORE
    # filing_date which is impossible in real life
    # We remove these bad records
    before = df.count()
    df = df.filter(
        f.col("date_of_filing").isNotNull() & (
            f.col("date_of_decision").isNull() |
            (f.col("date_of_decision") >=
             f.col("date_of_filing"))
        )
    )

    # WHY persist HERE, not before the filter:
    # Everything above this point (all 6 joins, date
    # casts, gender cleanup, fillna, the filter itself)
    # only needs to run ONCE now. Persisting earlier
    # meant this whole chain was recomputed a second
    # time during the final write, since the filter
    # created a new, uncached dataframe downstream of
    # the old cache point.
    #df = df.persist(StorageLevel.MEMORY_AND_DISK)

    after = df.count()
    print(f"Removed {before - after:,} impossible records")

    # ── STEP F: COMPUTE NEW FEATURE COLUMNS ──

    # is_resolved: 1 if case has decision date, 0 if pending
    df = df.withColumn("is_resolved",
        f.when(f.col("date_of_decision")
                .isNotNull(), 1).otherwise(0)
    )

    # case_age_days:
    # For resolved: days from filing to decision
    # For pending: days from filing to TODAY
    df = df.withColumn("case_age_days",
        f.when(
            f.col("date_of_decision").isNotNull(),
            f.datediff(f.col("date_of_decision"),
                       f.col("date_of_filing"))
        ).otherwise(
            f.datediff(f.current_date(),
                       f.col("date_of_filing"))
        )
    )

    # delay_category: classify case by how long it took
    df = df.withColumn("delay_category",
        f.when(f.col("case_age_days") < 365,   "Fast")
        .when(f.col("case_age_days") < 1095, "Medium")
        .when(f.col("case_age_days") < 2555,   "Slow")
        .otherwise("Stuck")
    )

    # filing_year: extract year from filing date
    # Used for year-wise trend charts
    df = df.withColumn("filing_year",
        f.year(f.col("date_of_filing"))
    )

    # hearing_span_days: days between first and last hearing
    # Shows how long the hearing process has been going on
    df = df.withColumn("hearing_span_days",
        f.when(
            f.col("date_first_list").isNotNull() &
            f.col("date_last_list").isNotNull(),
            f.datediff(f.col("date_last_list"),
                       f.col("date_first_list"))
        )
    )

    # ── STEP G: SELECT ONLY FINAL COLUMNS ──
    # WHY: We drop intermediate columns used for
    # joins (state_code, dist_code etc.) and keep
    # only the clean readable columns we need
    final_df = df.select(
        "ddl_case_id",
        f.col("state_name"),
        f.col("district_name"),
        f.col("court_name"),
        f.col("court_no"),
        f.col("judge_position"),
        f.col("type_name_s").alias("case_type"),
        f.col("purpose_name_s").alias("purpose"),
        f.col("disp_name_s").alias("disposal_type"),
        "date_of_filing",
        "date_of_decision",
        "date_first_list",
        "date_last_list",
        "date_next_list",
        "is_resolved",
        "case_age_days",
        "delay_category",
        "filing_year",
        "hearing_span_days",
        f.col("female_petitioner").alias("petitioner_gender"),
        f.col("female_defendant").alias("defendant_gender")
    )

    # ── STEP H: SAVE TO POSTGRESQL ──
    # WHY mode append:
    # Each batch ADDS to the table, not replaces it
    # So data from batch 1, 2, 3... all accumulates
    #
    # WHY repartition(4):
    # Controls how many parallel JDBC connections write
    # at once. With --executor-cores 1 this won't add
    # real parallelism yet, but costs nothing and is
    # ready to help the moment executor-cores is raised.
    #
    # WHY batchsize + rewriteBatchedInserts:
    # Spark's JDBC writer defaults to 1000 rows per
    # round-trip. batchsize=10000 sends far fewer,
    # larger round-trips. rewriteBatchedInserts is a
    # Postgres JDBC driver flag that rewrites batched
    # INSERTs into a single multi-row INSERT statement --
    # a large speedup for bulk writes into Postgres.
    final_df.repartition(4).write \
        .format("jdbc") \
        .option("url",
            "jdbc:postgresql://pipeline_postgres:5432/court_docket_db"
        ) \
        .option("dbtable", "cases_cleaned") \
        .option("user", "court_user") \
        .option("password", "court_password") \
        .option("driver", "org.postgresql.Driver") \
        .option("batchsize", 10000) \
        .option("rewriteBatchedInserts", "true") \
        .mode("append") \
        .save()

    # ── STEP I: LOG DATA QUALITY ──
    # Save quality metrics to data_quality_log table
    quality_data = spark.createDataFrame([{
        "total_received"    : before,
        "valid_records"     : after,
        "null_records"      : 0,
        "duplicate_records" : 0,
        "impossible_dates"  : before - after
    }])

    quality_data.write \
        .format("jdbc") \
        .option("url",
            "jdbc:postgresql://pipeline_postgres:5432/court_docket_db"
        ) \
        .option("dbtable", "data_quality_log") \
        .option("user", "court_user") \
        .option("password", "court_password") \
        .option("driver", "org.postgresql.Driver") \
        .mode("append") \
        .save()

    print(f"Batch {epoch_id}: {after:,} records saved to PostgreSQL")
    #df.unpersist()

# ─────────────────────────────────────────
# 6. START STREAMING
#
# WHY checkpointLocation:
# Spark saves its progress here.
# If job crashes and restarts, it continues
# from where it left off — no data lost,
# no data processed twice.
# NOTE: this path must be a mounted volume
# (see docker-compose.yml) or it resets on
# every container restart, forcing a full
# topic replay from earliest each time.
#
# WHY trigger(availableNow=True):
# Processes everything currently in Kafka in
# batches of maxOffsetsPerTrigger, then stops
# on its own -- good fit for Airflow-orchestrated
# batch-style runs rather than a forever-running
# streaming job.
# ─────────────────────────────────────────
print("Starting streaming pipeline...")

query = parsed.writeStream \
    .foreachBatch(process_batch) \
    .option("checkpointLocation",
            "/tmp/spark_checkpoint_court") \
    .trigger(availableNow=True) \
    .start()

print("Pipeline running. Waiting for Kafka messages...")
print("Press Ctrl+C to stop.")

query.awaitTermination()