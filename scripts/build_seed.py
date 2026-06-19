import argparse
import csv
import json
import os
import re
import sys
import unicodedata

from collections import Counter
from datetime import date, datetime, time, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = f"{PROJECT_DIR}/src"
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from elo import rating_update



BASE_RATING = 1500.0
HOME_ADVANTAGE = 75
UPDATE_HOME_ADVANTAGE = HOME_ADVANTAGE / 2
DEFAULT_START_DATE = "1930-01-01"
DEFAULT_END_DATE = "2026-06-11"
DEFAULT_SEED_YEARS = 4
DEFAULT_PRIOR_MATCHES = 30
CURRENT_NAME_OVERRIDES = {
    "United States": "USA",
    "United States of America": "USA",
    "United States Virgin Islands": "US Virgin Islands",
}


def parse_date(value):
    return date.fromisoformat(value)


def add_years(value, years):
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year + years)


def date_to_ts(value):
    return int(datetime.combine(value, time(12), timezone.utc).timestamp())


def sanitize_slug(name):
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.lower().replace("&", " and ")
    ascii_name = re.sub(r"[^a-z0-9]+", "-", ascii_name)
    return ascii_name.strip("-")


def parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_former_names(path):
    aliases = []
    if not os.path.isfile(path):
        return aliases

    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            aliases.append({
                "current": row["current"].strip(),
                "former": row["former"].strip(),
                "start_date": parse_date(row["start_date"]),
                "end_date": parse_date(row["end_date"]),
            })
    return aliases


def canonical_team(name, match_date, aliases):
    clean_name = name.strip()
    for alias in aliases:
        if (
            clean_name == alias["former"]
            and alias["start_date"] <= match_date <= alias["end_date"]
        ):
            return CURRENT_NAME_OVERRIDES.get(alias["current"], alias["current"])
    return CURRENT_NAME_OVERRIDES.get(clean_name, clean_name)


def load_matches(source_path, aliases, start_date, end_date):
    matches = []
    alias_counts = Counter()

    with open(source_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            match_date = parse_date(row["date"])
            if match_date < start_date or match_date >= end_date:
                continue

            home_name = canonical_team(row["home_team"], match_date, aliases)
            away_name = canonical_team(row["away_team"], match_date, aliases)
            if home_name != row["home_team"].strip():
                alias_counts[f"{row['home_team'].strip()} -> {home_name}"] += 1
            if away_name != row["away_team"].strip():
                alias_counts[f"{row['away_team'].strip()} -> {away_name}"] += 1

            matches.append({
                "date": match_date,
                "ts": date_to_ts(match_date),
                "homeSlug": sanitize_slug(home_name),
                "awaySlug": sanitize_slug(away_name),
                "homeName": home_name,
                "awayName": away_name,
                "hg": int(row["home_score"]),
                "ag": int(row["away_score"]),
                "leagueName": row["tournament"].strip(),
                "neutral": parse_bool(row["neutral"]),
            })

    matches.sort(key=lambda m: (m["date"], m["homeSlug"], m["awaySlug"]))
    return matches, alias_counts


def build_seed(matches, seed_start, seed_end, prior_matches):
    ratings = {}
    team_names = {}
    counts = Counter()
    seed_matches = 0

    for match in matches:
        if match["date"] < seed_start or match["date"] >= seed_end:
            continue

        home = match["homeSlug"]
        away = match["awaySlug"]
        ratings.setdefault(home, BASE_RATING)
        ratings.setdefault(away, BASE_RATING)
        team_names.setdefault(home, match["homeName"])
        team_names.setdefault(away, match["awayName"])

        home_advantage = 0 if match["neutral"] else UPDATE_HOME_ADVANTAGE
        delta = rating_update(
            ratings[home],
            ratings[away],
            match["hg"],
            match["ag"],
            match["leagueName"],
            ts=match["ts"],
            now_sec=match["ts"],
            home_advantage=home_advantage,
            use_recency=False,
        )

        ratings[home] += delta
        ratings[away] -= delta
        counts[home] += 1
        counts[away] += 1
        seed_matches += 1

    shrunk = {}
    details = {}
    for slug in sorted(ratings):
        n = counts[slug]
        reliability = n / (n + prior_matches)
        rating = BASE_RATING + reliability * (ratings[slug] - BASE_RATING)
        shrunk[slug] = round(rating)
        details[slug] = {
            "name": team_names[slug],
            "matches": n,
            "learnedRating": round(ratings[slug], 2),
            "reliability": round(reliability, 4),
        }

    return shrunk, details, seed_matches


def write_seed(args):
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    seed_end = add_years(start_date, args.years)

    aliases = load_former_names(args.aliases)
    matches, alias_counts = load_matches(args.source, aliases, start_date, end_date)
    ratings, teams, seed_matches = build_seed(
        matches,
        start_date,
        seed_end,
        args.prior_matches,
    )

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": os.path.relpath(args.source, PROJECT_DIR),
        "formerNames": os.path.relpath(args.aliases, PROJECT_DIR),
        "method": "elo_seed_from_first_history_window_with_sample_shrinkage",
        "baseRating": round(BASE_RATING),
        "priorMatches": args.prior_matches,
        "seedWindow": {
            "start": start_date.isoformat(),
            "endExclusive": seed_end.isoformat(),
            "years": args.years,
            "matches": seed_matches,
        },
        "calibrationWindow": {
            "start": seed_end.isoformat(),
            "endExclusive": end_date.isoformat(),
            "matchesAvailable": max(0, len(matches) - seed_matches),
        },
        "aliasesApplied": dict(sorted(alias_counts.items())),
        "ratings": ratings,
        "teams": teams,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")

    print(
        f"Wrote {args.output} with {len(ratings)} teams from "
        f"{seed_matches} seed matches ({start_date} to {seed_end})."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build initial Elo seed ratings from the first years of historical results."
    )
    parser.add_argument("--source", default=f"{PROJECT_DIR}/data/raw/results.csv")
    parser.add_argument("--aliases", default=f"{PROJECT_DIR}/data/reference/former_names.csv")
    parser.add_argument("--output", default=f"{PROJECT_DIR}/data/seeds/elo_seed.json")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--years", type=int, default=DEFAULT_SEED_YEARS)
    parser.add_argument("--prior-matches", type=int, default=DEFAULT_PRIOR_MATCHES)
    return parser.parse_args()


if __name__ == "__main__":
    write_seed(parse_args())
