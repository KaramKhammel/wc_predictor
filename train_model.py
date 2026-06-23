import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = f"{PROJECT_DIR}/src"
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

from ml_model import MatchModel, brier3, rps3
from ml_config import (
    FEATURES_FILE,
    MODEL_FILE,
    ML_REPORT_FILE,
    TEST_FRACTION,
)

META_COLS = ("y_out", "y_hg", "y_ag", "date", "ts", "league")


def _load_features(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path.rsplit(".", 1)[0] + ".csv")


def main():
    parser = argparse.ArgumentParser(
        description="Train the 1X2 + score model and benchmark it against the Elo engine."
    )
    parser.add_argument("--features", default=FEATURES_FILE)
    parser.add_argument("--out", default=MODEL_FILE)
    parser.add_argument("--report", default=ML_REPORT_FILE)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    args = parser.parse_args()

    df = _load_features(args.features)
    df = df.sort_values("ts").reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols]
    y_out = df["y_out"].to_numpy()
    y_hg = df["y_hg"].to_numpy()
    y_ag = df["y_ag"].to_numpy()

    # time-based split (no shuffling)
    split = int(len(df) * (1 - args.test_fraction))
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr_o, yte_o = y_out[:split], y_out[split:]
    ytr_hg, ytr_ag = y_hg[:split], y_ag[:split]
    yte_hg, yte_ag = y_hg[split:], y_ag[split:]

    print(f"train {len(Xtr):,} ({df['date'].iloc[0]} -> {df['date'].iloc[split-1]})")
    print(f"test  {len(Xte):,} ({df['date'].iloc[split]} -> {df['date'].iloc[-1]})")

    # ---- fit on TRAIN ONLY for the honest holdout report ----
    model = MatchModel(feature_names=feat_cols).fit(Xtr, ytr_o, ytr_hg, ytr_ag)

    # ---- evaluate ----
    ml_proba = model.predict_proba_clf(Xte)
    blend = model.predict_1x2(Xte)
    eng = Xte[["p_home", "p_draw", "p_away"]].to_numpy()
    eng = eng / eng.sum(axis=1, keepdims=True)
    lam_h, lam_a = model.predict_goals(Xte)

    def metrics(p):
        return dict(
            log_loss=float(log_loss(yte_o, p, labels=[0, 1, 2])),
            brier=brier3(p, yte_o),
            rps=rps3(p, yte_o),
            accuracy=float(accuracy_score(yte_o, p.argmax(1))),
        )

    report = {
        "test_matches": int(len(Xte)),
        "test_start": df["date"].iloc[split],
        "test_end": df["date"].iloc[-1],
        "elo_engine": metrics(eng),
        "ml_classifier": metrics(ml_proba),
        "blend": metrics(blend),
        "goals_mae": {
            "home_ml": float(mean_absolute_error(yte_hg, lam_h)),
            "home_engine": float(mean_absolute_error(yte_hg, Xte["exp_g_home"])),
            "away_ml": float(mean_absolute_error(yte_ag, lam_a)),
            "away_engine": float(mean_absolute_error(yte_ag, Xte["exp_g_away"])),
        },
    }
    # exact scoreline accuracy from the Poisson grid
    exact = 0
    for i in range(len(Xte)):
        M = model.score_matrix(lam_h[i], lam_a[i])
        ph, pa = np.unravel_index(M.argmax(), M.shape)
        exact += int(ph == yte_hg[i] and pa == yte_ag[i])
    report["exact_scoreline_accuracy"] = exact / len(Xte)

    # ---- print comparison table ----
    print("\n  1X2 metrics (lower better except accuracy):")
    print(f"  {'':14}{'log-loss':>10}{'brier':>9}{'rps':>9}{'acc':>8}")
    for name, key in [("Elo engine", "elo_engine"),
                      ("ML model", "ml_classifier"),
                      ("Blend", "blend")]:
        m = report[key]
        print(f"  {name:<14}{m['log_loss']:>10.4f}{m['brier']:>9.4f}"
              f"{m['rps']:>9.4f}{m['accuracy']:>8.4f}")
    print(f"\n  expected-goals MAE  home: engine {report['goals_mae']['home_engine']:.3f}"
          f" | ml {report['goals_mae']['home_ml']:.3f}")
    print(f"  expected-goals MAE  away: engine {report['goals_mae']['away_engine']:.3f}"
          f" | ml {report['goals_mae']['away_ml']:.3f}")
    print(f"  exact-scoreline accuracy: {report['exact_scoreline_accuracy']*100:.1f}%")

    # ---- refit on FULL history for the deployable model, then save ----
    print("\n  refitting on full history for deployment...")
    final = MatchModel(feature_names=feat_cols).fit(X, y_out, y_hg, y_ag)
    final.save(args.out)
    print(f"  model saved -> {args.out}")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  report saved -> {args.report}")


if __name__ == "__main__":
    main()
