import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = f"{PROJECT_DIR}/src"
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd

from features import build_training_table
from ml_config import FEATURES_FILE


def main():
    parser = argparse.ArgumentParser(
        description="Build the leakage-free walk-forward feature table for the ML model."
    )
    parser.add_argument("--out", default=FEATURES_FILE,
                        help="Output path for the feature table (.parquet).")
    parser.add_argument("--recency", action="store_true",
                        help="Use Elo recency decay when building features.")
    args = parser.parse_args()

    print("building walk-forward features (this reuses the Elo engine match-by-match)...")
    rows, y_out, y_hg, y_ag, meta, _, _ = build_training_table(use_recency=args.recency)

    df = pd.DataFrame(rows)
    df["y_out"] = y_out
    df["y_hg"] = y_hg
    df["y_ag"] = y_ag
    df["date"] = [m["date"] for m in meta]
    df["ts"] = [m["ts"] for m in meta]
    df["league"] = [m["league"] for m in meta]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    try:
        df.to_parquet(args.out, index=False)
    except Exception:
        # parquet engine not installed -> fall back to csv
        args.out = args.out.rsplit(".", 1)[0] + ".csv"
        df.to_csv(args.out, index=False)

    n_feat = len([c for c in df.columns
                  if c not in ("y_out", "y_hg", "y_ag", "date", "ts", "league")])
    print(f"  rows: {len(df):,}   features: {n_feat}")
    print(f"  date range: {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
