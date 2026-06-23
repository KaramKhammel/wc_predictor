"""
ML configuration: paths and hyperparameters for the learned outcome/score model.

Follows the same convention as config.py (env-overridable paths) so the ML layer
plugs into the existing project layout without touching the Elo engine.
"""
import os

from config import (
    OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    HOME_ADVANTAGE,
)

# ---- artifact locations ----------------------------------------------------------
MODELS_OUTPUT_DIR = os.environ.get("WC_MODELS_OUTPUT_DIR", f"{OUTPUT_DIR}/models")

# feature table built by build_features.py (consumed by train_model.py)
FEATURES_FILE = os.environ.get(
    "WC_FEATURES_FILE", f"{PROCESSED_DATA_DIR}/ml_features.parquet"
)

# trained model bundle written by train_model.py (consumed by predict_ml.py)
MODEL_FILE = os.environ.get("WC_MODEL_FILE", f"{MODELS_OUTPUT_DIR}/match_model.joblib")

# training report (metrics vs the Elo baseline)
ML_REPORT_FILE = os.environ.get(
    "WC_ML_REPORT_FILE", f"{OUTPUT_DIR}/reports/ml_backtest.json"
)

# ---- feature engineering ---------------------------------------------------------
FORM_WINDOWS = (5, 10, 15)        # rolling windows requested for form / gd / xg-conv
BURN_IN_MATCHES = 5000            # skip the earliest matches (sparse rolling history)
MAX_GOALS = 10                    # scoreline grid size for the Poisson score model

# ---- model hyperparameters -------------------------------------------------------
# 1X2 outcome classifier (XGBoost multi:softprob)
CLF_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_jobs=4,
    random_state=42,
)

# goal regressors (gradient-boosted Poisson, home and away)
GOAL_REG_PARAMS = dict(
    loss="poisson",
    max_iter=500,
    learning_rate=0.03,
    max_depth=4,
    l2_regularization=2.0,
    random_state=42,
)

# time-based holdout fraction for the train script's report
TEST_FRACTION = 0.15

# blend weight: final 1X2 = BLEND_W * ML + (1 - BLEND_W) * Elo-engine probs
BLEND_W = 0.6

# default seed for prediction when a team is unknown
DEFAULT_RATING = 1500
