import json
import argparse

from elo import expected_goals, match_prob, predicted_scoreline
from config import OUTPUT_DIR


with open(f"{OUTPUT_DIR}/elo-calibrated.json", "r", encoding="utf-8") as f:
        ratings = json.load(f)["ratings"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('home')
    parser.add_argument('away')
    parser.add_argument('--home-advantage', type=float, default=0,
                        help="Elo points of home advantage (0 = neutral venue).")
    args = parser.parse_args()

    home_rating = ratings.get(args.home, 1500)
    away_rating = ratings.get(args.away, 1500)
    ha = args.home_advantage

    pA, pD, pB = match_prob(home_rating, away_rating, ha)
    scoreline = predicted_scoreline(home_rating, away_rating, ha)

    venue = "   [neutral]" if ha == 0 else f"   [home adv {ha:g}]"
    print(f"\n  {args.home} (Elo {home_rating})  vs  {args.away} (Elo {away_rating}){venue}\n")
    print(f"  {args.home.ljust(16)} win  {pA * 100:5.1f}%  {'█' * round(pA * 30)}")
    print(f"  {'draw'.ljust(16)}      {pD * 100:5.1f}%  {'█' * round(pD * 30)}")
    print(f"  {args.away.ljust(16)} win  {pB * 100:5.1f}%  {'█' * round(pB * 30)}\n")
    print(f"  expected goals:  {expected_goals(home_rating, away_rating):.2f} – {expected_goals(away_rating, home_rating):.2f}\n")
    print(f"  predicted scoreline:  {scoreline[0]} – {scoreline[1]} (probability {scoreline[2] * 100:.1f}%)\n")


