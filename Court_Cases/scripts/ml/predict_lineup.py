import io
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


# WHY pipeline_postgres, not localhost: same reason
# as train_model.py -- this runs inside the Airflow
# container, on the Docker network.
engine = create_engine(
    "postgresql://court_user:court_password"
    "@pipeline_postgres:5432/court_docket_db"
)

# Load models from PostgreSQL
print("Loading models...")

model_row = pd.read_sql("""
    SELECT model_binary, reg_model_binary, encoder_binary
    FROM model_registry
    ORDER BY trained_on DESC
    LIMIT 1
""", engine)

if model_row.empty:
    raise RuntimeError(
        "No trained model found in model_registry. "
        "Run train_model.py before predict_lineup.py."
    )

cls_model = pickle.load(
    io.BytesIO(model_row['model_binary'].iloc[0])
)
# WHY reg_model now loads from Postgres too, instead of
# a local .pkl file: removes the fragile dependency on
# train_model.py and predict_lineup.py sharing the same
# container filesystem / not being recreated in between.
reg_model = pickle.load(
    io.BytesIO(model_row['reg_model_binary'].iloc[0])
)
encoders = pickle.load(
    io.BytesIO(model_row['encoder_binary'].iloc[0])
)
le_judge = encoders['judge']
le_case  = encoders['case']

print("Models loaded")

# Load pending cases
print("Loading pending cases...")

query = """
    SELECT
        ddl_case_id,
        state_name,
        district_name,
        case_type,
        date_of_filing,
        date_next_list,
        case_age_days,
        delay_category,
        petitioner_gender,
        disposal_type,
        judge_position,
        hearing_span_days,
        filing_year
    FROM cases_cleaned
    WHERE is_resolved = 0
    AND date_next_list IS NOT NULL
"""

first = True
total_predictions = 0
dispose_count = 0
adjourn_count = 0

print("Generating predictions...")

# known_judges = set(le_judge.classes_)
# known_cases = set(le_case.classes_)


judge_map = {cls: idx for idx, cls in enumerate(le_judge.classes_)}
case_map  = {cls: idx for idx, cls in enumerate(le_case.classes_)}

FEATURES = [
    'filing_year',
    'filing_month',
    'filing_day_of_week',
    'hearing_span_days',
    'delay_cat_encoded',
    'gender_encoded',
    'judge_encoded',
    'case_type_encoded'
]

lineup_cols = [
    'ddl_case_id',
    'state_name',
    'district_name',
    'case_type',
    'date_of_filing',
    'date_next_list',
    'case_age_days',
    'delay_category',
    'petitioner_gender',
    'disposal_type',
    'resolution_probability',
    'predicted_outcome',
    'predicted_days_remaining',
    'prediction_date'
]

# WHY a separate streaming connection: stream_results=True
# triggers a Postgres server-side (named) cursor, which only
# supports SELECT statements. Applying it to the whole engine
# broke the CREATE TABLE issued later by to_sql(). Scoping it
# to just this one connection keeps the SELECT streaming while
# leaving to_sql()'s writes (which use the plain `engine`)
# unaffected.
with engine.connect().execution_options(stream_results=True) as stream_conn:
    for df in pd.read_sql(query, stream_conn, chunksize=50000):

        print(f"Processing {len(df):,} rows...")

        # Feature engineering
        df['date_of_filing'] = pd.to_datetime(df['date_of_filing'])
        df['filing_month'] = df['date_of_filing'].dt.month
        df['filing_day_of_week'] = df['date_of_filing'].dt.dayofweek

        df['delay_cat_encoded'] = df['delay_category'].map({
            'Fast': 0,
            'Medium': 1,
            'Slow': 2,
            'Stuck': 3
        })

        df['gender_encoded'] = df['petitioner_gender'].map({
            'female': 1,
            'male': 0,
            'unknown': -1
        })

        df['judge_encoded'] = df['judge_position'].fillna('unknown').map(judge_map).fillna(-1).astype(int)
        df['case_type_encoded'] = df['case_type'].fillna('unknown').map(case_map).fillna(-1).astype(int)

        X = df[FEATURES].fillna(0)

        # Predictions
        df['resolution_probability'] = cls_model.predict_proba(X)[:, 1]

        df['predicted_outcome'] = np.where(
            df['resolution_probability'] >= 0.5,
            'DISPOSE',
            'ADJOURN'
        )

        df['predicted_days_remaining'] = reg_model.predict(X)
        df['prediction_date'] = pd.Timestamp.today().date()

        df[lineup_cols].to_sql(
            'daily_lineup',
            engine,          # ← plain engine, not stream_conn
            if_exists='replace' if first else 'append',
            index=False
        )

        first = False

        total_predictions += len(df)
        dispose_count += (df['predicted_outcome'] == 'DISPOSE').sum()
        adjourn_count += (df['predicted_outcome'] == 'ADJOURN').sum()

        del df, X

print(f"\nPredictions saved: {total_predictions:,} cases")
print(f"Likely disposed today : {dispose_count:,}")
print(f"Likely adjourned today: {adjourn_count:,}")
