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

from ml_model import MatchModel
from features import build_match_features, build_training_table
from ml_config import MODEL_FILE, DEFAULT_RATING


def print_score_matrix(matrix, home, away, display_goals):
    max_goals = min(display_goals, len(matrix) - 1)
    print("  score matrix: exact-score probability")
    print(f"  rows = {home} goals, columns = {away} goals\n")
    header = "      " + " ".join(f"{g:>6}" for g in range(max_goals + 1))
    print(header)
    for home_goals in range(max_goals + 1):
        row = matrix[home_goals]
        cells = " ".join(f"{p * 100:5.1f}%" for p in row[:max_goals + 1])
        print(f"  {home_goals:>2}  {cells}")
    print()


def top_scorelines(matrix, limit):
    scores = []
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            scores.append((float(matrix[home_goals, away_goals]), home_goals, away_goals))
    scores.sort(reverse=True)
    return scores[:limit]


def main():
    parser = argparse.ArgumentParser(
        description="Predict a single match with the trained ML model "
                    "(same output as predict.py)."
    )
    parser.add_argument("home")
    parser.add_argument("away")
    parser.add_argument("--home-advantage", type=float, default=0,
                        help="Elo points of home advantage (0 = neutral venue).")
    parser.add_argument("--matrix-goals", type=int, default=5,
                        help="Highest goal count to show in the exact-score matrix.")
    parser.add_argument("--top-scores", type=int, default=8,
                        help="Number of most likely exact scores to list.")
    parser.add_argument("--model", default=MODEL_FILE)
    args = parser.parse_args()

    model = MatchModel.load(args.model)

    # Rebuild CURRENT engine state (ratings + each team's rolling history) by walking
    # the full history once. This gives the live features the model expects.
    print("  loading engine state (walking history)...", flush=True)
    _, _, _, _, _, ratings, histories = build_training_table(burn_in=10 ** 9)

    home_rating = ratings.get(args.home, DEFAULT_RATING)
    away_rating = ratings.get(args.away, DEFAULT_RATING)
    ha = args.home_advantage

    home_hist = histories.get(args.home, [])
    away_hist = histories.get(args.away, [])

    feat = build_match_features(home_rating, away_rating, ha,
                                home_hist, away_hist, "Friendly")
    X = pd.DataFrame([feat])

    # 1X2 (blended ML + engine) and goal rates -> score matrix
    pA, pD, pB = model.predict_1x2(X)[0]
    lam_h, lam_a = model.predict_goals(X)
    lam_h, lam_a = float(lam_h[0]), float(lam_a[0])
    matrix = model.score_matrix(lam_h, lam_a)

    top_scores = top_scorelines(matrix, args.top_scores)
    most_likely_p, ml_hg, ml_ag = top_scores[0]

    # best score consistent with the top 1X2 outcome (mirrors predict.py's scoreline)
    outcome = int(np.argmax([pA, pD, pB]))
    idx = np.arange(matrix.shape[0])
    if outcome == 0:      # home win -> home goals > away goals
        mask = np.greater.outer(idx, idx)
    elif outcome == 1:    # draw
        mask = np.equal.outer(idx, idx)
    else:                 # away win
        mask = np.less.outer(idx, idx)
    masked = np.where(mask, matrix, -1.0)
    best_hg, best_ag = np.unravel_index(masked.argmax(), masked.shape)
    best_p = float(matrix[best_hg, best_ag])

    venue = "   [neutral]" if ha == 0 else f"   [home adv {ha:g}]"
    print(f"\n  {args.home} (Elo {home_rating:.0f})  vs  {args.away} (Elo {away_rating:.0f}){venue}  [ML model]\n")
    print(f"  {args.home.ljust(16)} win  {pA * 100:5.1f}%  {'█' * round(pA * 30)}")
    print(f"  {'draw'.ljust(16)}      {pD * 100:5.1f}%  {'█' * round(pD * 30)}")
    print(f"  {args.away.ljust(16)} win  {pB * 100:5.1f}%  {'█' * round(pB * 30)}\n")
    print(f"  expected goals:  {lam_h:.2f} - {lam_a:.2f}\n")
    print(f"  most likely exact score:  {ml_hg} - {ml_ag} (probability {most_likely_p * 100:.1f}%)")
    print(f"  best score for top 1X2 outcome:  {best_hg} - {best_ag} (probability {best_p * 100:.1f}%)\n")
    print_score_matrix(matrix, args.home, args.away, args.matrix_goals)

    print("  most likely exact scores:")
    for probability, home_goals, away_goals in top_scores:
        print(f"    {home_goals}-{away_goals}: {probability * 100:4.1f}%")
    print()


if __name__ == "__main__":
    main()
