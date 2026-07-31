import os
import pandas as pd
from sqlalchemy import create_engine

# ============================================================
# PostgreSQL Connection
# ============================================================

engine = create_engine(
    "postgresql://court_user:court_password@localhost:5432/court_docket_db"
)

# ============================================================
# Output Folder
# ============================================================

OUTPUT_DIR = "dashboard_exports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Small / Aggregated Tables
# ============================================================

summary_tables = [
    "yearly_summary",
    "state_summary",
    "district_summary",
    "casetype_summary",
    "delay_summary",
    "judge_summary",
    "gender_summary",
    "time_trends",
    "disposal_summary",
    "hearing_span_summary",
    "defendant_gender_summary",
    "oldest_pending_cases",
    "casetype_delay_crosstab"
]

print("=" * 60)
print("Exporting dashboard summary tables")
print("=" * 60)

for table in summary_tables:

    print(f"\nExporting {table}...")

    df = pd.read_sql(f"SELECT * FROM {table}", engine)

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{table}.csv"
    )

    df.to_csv(output_file, index=False)

    print(f"✓ Saved {output_file} ({len(df):,} rows)")

print("\nAll summary tables exported successfully.")

# ============================================================
# Export Daily Lineup (Chunked)
# ============================================================

print("\n" + "=" * 60)
print("Exporting daily_lineup")
print("=" * 60)

query = "SELECT * FROM daily_lineup"

chunks = pd.read_sql(
    query,
    engine,
    chunksize=100000
)

output_file = os.path.join(
    OUTPUT_DIR,
    "daily_lineup.csv"
)

first = True
total_rows = 0

for chunk in chunks:

    chunk.to_csv(
        output_file,
        mode="w" if first else "a",
        header=first,
        index=False
    )

    total_rows += len(chunk)

    print(f"Written {total_rows:,} rows...")

    first = False

print(f"\n✓ Saved daily_lineup.csv ({total_rows:,} rows)")

# ============================================================
# OPTIONAL
# Export a manageable sample of raw data
# (Useful for Power BI drill-through)
# ============================================================

# print("\n" + "=" * 60)
# print("Exporting sample of cases_cleaned")
# print("=" * 60)

# sample_query = """
# SELECT *
# FROM cases_cleaned
# TABLESAMPLE SYSTEM (1)
# """

# sample = pd.read_sql(sample_query, engine)

# sample_file = os.path.join(
#     OUTPUT_DIR,
#     "cases_cleaned_sample.csv"
# )

# sample.to_csv(sample_file, index=False)

# print(f"✓ Saved cases_cleaned_sample.csv ({len(sample):,} rows)")

# ============================================================
# Cleanup
# ============================================================

engine.dispose()

print("\n" + "=" * 60)
print("EXPORT COMPLETE")
print("=" * 60)

print(f"\nFiles saved to:\n{os.path.abspath(OUTPUT_DIR)}")