import json
import argparse

from elo import MAX_GOALS, match_prob, predicted_scoreline, score_matrix
from config import CALIBRATED_RATINGS_FILE


with open(CALIBRATED_RATINGS_FILE, "r", encoding="utf-8") as f:
        ratings = json.load(f)["ratings"]


def print_score_matrix(matrix, home, away, display_goals):
    max_goals = min(display_goals, len(matrix) - 1)
    print("  score matrix: exact-score probability")
    print(f"  rows = {home} goals, columns = {away} goals\n")

    header = "      " + " ".join(f"{g:>6}" for g in range(max_goals + 1))
    print(header)
    for home_goals, row in enumerate(matrix[:max_goals + 1]):
        cells = " ".join(f"{p * 100:5.1f}%" for p in row[:max_goals + 1])
        print(f"  {home_goals:>2}  {cells}")
    print()


def top_scorelines(matrix, limit):
    scores = []
    for home_goals, row in enumerate(matrix):
        for away_goals, probability in enumerate(row):
            scores.append((probability, home_goals, away_goals))
    scores.sort(reverse=True)
    return scores[:limit]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('home')
    parser.add_argument('away')
    parser.add_argument('--home-advantage', type=float, default=0,
                        help="Elo points of home advantage (0 = neutral venue).")
    parser.add_argument('--matrix-goals', type=int, default=5,
                        help="Highest goal count to show in the exact-score matrix.")
    parser.add_argument('--top-scores', type=int, default=8,
                        help="Number of most likely exact scores to list.")
    args = parser.parse_args()

    home_rating = ratings.get(args.home, 1500)
    away_rating = ratings.get(args.away, 1500)
    ha = args.home_advantage

    pA, pD, pB = match_prob(home_rating, away_rating, ha)
    scoreline = predicted_scoreline(home_rating, away_rating, ha)
    matrix, xg_home, xg_away = score_matrix(
        home_rating,
        away_rating,
        ha,
        max_goals=max(MAX_GOALS, args.matrix_goals),
    )

    venue = "   [neutral]" if ha == 0 else f"   [home adv {ha:g}]"
    print(f"\n  {args.home} (Elo {home_rating})  vs  {args.away} (Elo {away_rating}){venue}\n")
    print(f"  {args.home.ljust(16)} win  {pA * 100:5.1f}%  {'█' * round(pA * 30)}")
    print(f"  {'draw'.ljust(16)}      {pD * 100:5.1f}%  {'█' * round(pD * 30)}")
    print(f"  {args.away.ljust(16)} win  {pB * 100:5.1f}%  {'█' * round(pB * 30)}\n")
    print(f"  expected goals:  {xg_home:.2f} – {xg_away:.2f}\n")
    # print(f"  predicted scoreline:  {scoreline[0]} – {scoreline[1]} (probability {scoreline[2] * 100:.1f}%)\n")
    print_score_matrix(matrix, args.home, args.away, args.matrix_goals)

    print("  most likely exact scores:")
    for probability, home_goals, away_goals in top_scorelines(matrix, args.top_scores):
        print(f"    {home_goals}-{away_goals}: {probability * 100:4.1f}%")
    print()


