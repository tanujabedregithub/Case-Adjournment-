import io
import os
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sqlalchemy import create_engine
from sqlalchemy.types import LargeBinary
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc

# ─────────────────────────────────────────
# WHY pipeline_postgres, not localhost:
# This runs as a python3 process inside the
# Airflow container, on the same Docker network
# as Postgres. "localhost" would only work if
# run directly on the host machine.
# ─────────────────────────────────────────
engine = create_engine(
    "postgresql://court_user:court_password"
    "@pipeline_postgres:5432/court_docket_db"
)

# WHY this directory: container filesystems are
# ephemeral. This path should be a mounted volume
# (see docker-compose.yml changes below) so pkl/png
# outputs survive container restarts and are visible
# on the host for inspection.
OUTPUT_DIR = "/opt/airflow/data/ml_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── TEMPORAL SPLIT LOADING ──
print("Loading data with temporal split...")

def load_year(year):
    query = f"""
        SELECT
            ddl_case_id,
            case_age_days,
            filing_year,
            hearing_span_days,
            delay_category,
            judge_position,
            case_type,
            petitioner_gender,
            date_of_filing,
            is_resolved
        FROM cases_cleaned
        WHERE filing_year = {year}
        LIMIT 100000
    """

    return pd.read_sql(query, engine)

# Temporal split
# 2010-2015 → train
# 2016-2017 → validation
# 2018      → test
train_frames = []
val_frames   = []
test_frames  = []

years_df = pd.read_sql("""
    SELECT DISTINCT filing_year
    FROM cases_cleaned
    WHERE filing_year IS NOT NULL
    ORDER BY filing_year
""", engine)

years = years_df['filing_year'].tolist()
print(f"Years available: {years}")

for year in years:
    df_year = load_year(year)

    if year <= 2015:
        train_frames.append(df_year)
    elif year <= 2017:
        val_frames.append(df_year)
    else:
        test_frames.append(df_year)

    print(f"Year {year}: {len(df_year):,} rows")

train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
val_df   = pd.concat(val_frames,   ignore_index=True) if val_frames   else pd.DataFrame()
test_df  = pd.concat(test_frames,  ignore_index=True) if test_frames  else pd.DataFrame()

# If not enough years, fallback to train_test_split
if len(val_df) == 0 or len(test_df) == 0:
    print("Not enough years for temporal split — using random split")
    all_df   = pd.concat(train_frames, ignore_index=True)
    train_df, temp_df = train_test_split(all_df, test_size=0.3, random_state=42)
    val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

del train_frames, val_frames, test_frames
gc.collect()

print(f"\nTrain: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# ── FEATURE ENGINEERING ──
def engineer_features(df):
    df = df.copy()
    df['date_of_filing']     = pd.to_datetime(df['date_of_filing'])
    df['filing_month']       = df['date_of_filing'].dt.month
    df['filing_day_of_week'] = df['date_of_filing'].dt.dayofweek
    df['delay_cat_encoded']  = df['delay_category'].map({
        'Fast': 0, 'Medium': 1, 'Slow': 2, 'Stuck': 3
    })
    df['gender_encoded'] = df['petitioner_gender'].map({
        'female': 1, 'male': 0, 'unknown': -1
    })
    return df

train_df = engineer_features(train_df)
val_df   = engineer_features(val_df)
test_df  = engineer_features(test_df)

# Fit encoders on training data only
le_judge = LabelEncoder()
le_case  = LabelEncoder()

train_df['judge_encoded'] = le_judge.fit_transform(
    train_df['judge_position'].fillna('unknown')
)
train_df['case_type_encoded'] = le_case.fit_transform(
    train_df['case_type'].fillna('unknown')
)

def encode_unseen(series, encoder):
    known = set(encoder.classes_)
    return series.apply(
        lambda x: encoder.transform([x])[0]
        if x in known else -1
    )

for df in [val_df, test_df]:
    df['judge_encoded'] = encode_unseen(
        df['judge_position'].fillna('unknown'), le_judge
    )
    df['case_type_encoded'] = encode_unseen(
        df['case_type'].fillna('unknown'), le_case
    )

FEATURES = [
    'filing_year', 'filing_month', 'filing_day_of_week',
    'hearing_span_days', 'delay_cat_encoded',
    'gender_encoded', 'judge_encoded', 'case_type_encoded'
]

X_train = train_df[FEATURES].fillna(0)
y_cls_train = train_df['is_resolved']
y_reg_train = train_df.loc[
    train_df['is_resolved'] == 1, 'case_age_days'
]
X_reg_train = train_df.loc[
    train_df['is_resolved'] == 1, FEATURES
].fillna(0)

X_val   = val_df[FEATURES].fillna(0)
y_cls_val = val_df['is_resolved']

X_test  = test_df[FEATURES].fillna(0)
y_cls_test = test_df['is_resolved']

print(f"\nClass distribution (train):\n{y_cls_train.value_counts()}")

# ── SMOTE ──
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(
    X_train, y_cls_train
)
print(f"After SMOTE: {dict(pd.Series(y_train_bal).value_counts())}")

# ── CLASSIFICATION MODELS ──
cls_results = []

def evaluate_cls(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)[:, 1]
    result = {
        'model_name': name,
        'accuracy'  : round(accuracy_score(y_te, pred), 4),
        'precision' : round(precision_score(y_te, pred, zero_division=0), 4),
        'recall'    : round(recall_score(y_te, pred, zero_division=0), 4),
        'f1_score'  : round(f1_score(y_te, pred, zero_division=0), 4),
        'roc_auc'   : round(roc_auc_score(y_te, prob), 4),
        'mae'       : None,
        'rmse'      : None,
        'r2_score'  : None
    }
    print(f"{name}: F1={result['f1_score']} ROC-AUC={result['roc_auc']}")
    return model, result

print("\n── Classification Models ──")

lr_model, lr_res = evaluate_cls(
    'Logistic Regression',
    LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    X_train_bal, y_train_bal, X_test, y_cls_test
)
cls_results.append(lr_res)

rf_model, rf_res = evaluate_cls(
    'Random Forest',
    RandomForestClassifier(
        n_estimators=50, max_depth=10,
        random_state=42, n_jobs=-1
    ),
    X_train_bal, y_train_bal, X_test, y_cls_test
)
cls_results.append(rf_res)

xgb_model, xgb_res = evaluate_cls(
    'XGBoost',
    XGBClassifier(
        n_estimators=100, learning_rate=0.05,
        max_depth=6, random_state=42,
        n_jobs=-1, verbosity=0,
        eval_metric='logloss'
    ),
    X_train_bal, y_train_bal, X_test, y_cls_test
)
cls_results.append(xgb_res)

lgbm_cls, lgbm_res = evaluate_cls(
    'LightGBM',
    LGBMClassifier(
        n_estimators=100, learning_rate=0.05,
        num_leaves=31, random_state=42,
        n_jobs=-1, is_unbalance=True, verbosity=-1
    ),
    X_train_bal, y_train_bal, X_test, y_cls_test
)
cls_results.append(lgbm_res)

# ── REGRESSION MODELS ──
print("\n── Regression Models (Days to Resolve) ──")

reg_results = []

def evaluate_reg(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mae  = round(mean_absolute_error(y_te, pred), 2)
    rmse = round(np.sqrt(mean_squared_error(y_te, pred)), 2)
    r2   = round(r2_score(y_te, pred), 4)
    result = {
        'model_name': f"{name} (Regression)",
        'accuracy'  : None,
        'precision' : None,
        'recall'    : None,
        'f1_score'  : None,
        'roc_auc'   : None,
        'mae'       : mae,
        'rmse'      : rmse,
        'r2_score'  : r2
    }
    print(f"{name}: MAE={mae} RMSE={rmse} R²={r2}")
    return model, result

X_reg_test = test_df.loc[
    test_df['is_resolved'] == 1, FEATURES
].fillna(0)
y_reg_test = test_df.loc[
    test_df['is_resolved'] == 1, 'case_age_days'
]

xgb_reg, xgb_reg_res = evaluate_reg(
    'XGBoost',
    XGBRegressor(
        n_estimators=100, learning_rate=0.05,
        max_depth=6, random_state=42,
        n_jobs=-1, verbosity=0
    ),
    X_reg_train, y_reg_train,
    X_reg_test, y_reg_test
)
reg_results.append(xgb_reg_res)

lgbm_reg, lgbm_reg_res = evaluate_reg(
    'LightGBM',
    LGBMRegressor(
        n_estimators=100, learning_rate=0.05,
        num_leaves=31, random_state=42,
        n_jobs=-1, verbosity=-1
    ),
    X_reg_train, y_reg_train,
    X_reg_test, y_reg_test
)
reg_results.append(lgbm_reg_res)

# ── SAVE ALL RESULTS ──
all_results = pd.DataFrame(cls_results + reg_results)

all_results.to_sql(
    'model_comparison', engine,
    if_exists='replace', index=False
)
print("\nModel comparison saved")

# Feature importance
importance_df = pd.DataFrame({
    'feature'   : FEATURES,
    'importance': lgbm_cls.feature_importances_
}).sort_values('importance', ascending=False)

importance_df.to_sql(
    'feature_importance', engine,
    if_exists='replace', index=False
)
print("Feature importance saved")

# SHAP
print("\nComputing SHAP...")
explainer   = shap.TreeExplainer(lgbm_cls)
shap_values = explainer.shap_values(X_test[:200])

shap_array = shap_values[1] \
    if isinstance(shap_values, list) else shap_values

shap_df = pd.DataFrame(
    shap_array,
    columns=[f'shap_{c}' for c in FEATURES]
)
shap_df.to_sql(
    'shap_values', engine,
    if_exists='replace', index=False
)

shap.summary_plot(shap_values, X_test[:200],
                  feature_names=FEATURES, show=False)
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'), bbox_inches='tight')
plt.close()
print("SHAP saved")

# ─────────────────────────────────────────
# Save models to PostgreSQL
#
# WHY LargeBinary dtype is REQUIRED here:
# Without it, pandas infers a text column for
# these bytes objects, which can corrupt the
# pickled binary on write (or fail outright,
# depending on driver versions). This is the
# single most important fix in this script.
#
# WHY reg_model_binary is now included:
# The original script only stored cls_model in
# Postgres and left reg_model as a local pickle
# file. That's fragile -- if train_model.py and
# predict_lineup.py ever run in different
# containers, or this container gets recreated,
# the local file is gone. Storing both models
# the same way removes that dependency entirely.
# ─────────────────────────────────────────
cls_buffer = io.BytesIO()
reg_buffer = io.BytesIO()
enc_buffer = io.BytesIO()

pickle.dump(lgbm_cls, cls_buffer)
pickle.dump(lgbm_reg, reg_buffer)
pickle.dump({'judge': le_judge, 'case': le_case}, enc_buffer)

pd.DataFrame([{
    'model_name'       : 'LightGBM',
    'model_version'    : '1.0',
    'trained_on'       : pd.Timestamp.now(),
    'f1_score'         : lgbm_res['f1_score'],
    'roc_auc'          : lgbm_res['roc_auc'],
    'accuracy'         : lgbm_res['accuracy'],
    'mae'              : lgbm_reg_res['mae'],
    'rmse'             : lgbm_reg_res['rmse'],
    'model_binary'     : cls_buffer.getvalue(),
    'reg_model_binary' : reg_buffer.getvalue(),
    'encoder_binary'   : enc_buffer.getvalue()
}]).to_sql(
    'model_registry', engine,
    if_exists='replace', index=False,
    dtype={
        'model_binary'    : LargeBinary,
        'reg_model_binary': LargeBinary,
        'encoder_binary'  : LargeBinary
    }
)
print("Models saved to model_registry (Postgres)")

# Local backup (optional -- kept for convenience when
# working directly inside this container, but the
# prediction script no longer depends on these files)
with open(os.path.join(OUTPUT_DIR, 'court_lgbm_cls.pkl'), 'wb') as f:
    pickle.dump(lgbm_cls, f)
with open(os.path.join(OUTPUT_DIR, 'court_lgbm_reg.pkl'), 'wb') as f:
    pickle.dump(lgbm_reg, f)
with open(os.path.join(OUTPUT_DIR, 'encoders.pkl'), 'wb') as f:
    pickle.dump({'judge': le_judge, 'case': le_case}, f)

print("\n========== DONE ==========")
print(f"Train: {len(train_df):,}")
print(f"Val  : {len(val_df):,}")
print(f"Test : {len(test_df):,}")
print(f"Best CLS F1  : {lgbm_res['f1_score']}")
print(f"Best REG MAE : {lgbm_reg_res['mae']}")
print("==========================")