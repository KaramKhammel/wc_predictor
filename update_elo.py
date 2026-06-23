import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, time, timezone


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = f"{PROJECT_DIR}/src"
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from elo import rating_update


HOSTS = ("mexico", "usa", "canada")
HOME_ADVANTAGE = 75
UPDATE_HOME_ADVANTAGE = HOME_ADVANTAGE / 2
LEAGUE_NAME = "World Cup"
DEFAULT_RATINGS_FILE = f"{PROJECT_DIR}/output/ratings/elo_calibrated.json"
DEFAULT_RESULTS_FILE = f"{PROJECT_DIR}/data/raw/wc2026_results.json"


def sanitize_slug(value):
    if not value:
        return None
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_value = ascii_value.lower().replace("&", " and ")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value)
    return slug.strip("-")


def date_to_ts(date_str):
    return int(
        datetime.fromisoformat(date_str)
        .replace(hour=12, tzinfo=timezone.utc)
        .timestamp()
    )


def infer_world_cup_home_side(match):
    team1 = sanitize_slug(match.get("t1") or match["team1"])
    team2 = sanitize_slug(match.get("t2") or match["team2"])
    g1 = int(match["g1"])
    g2 = int(match["g2"])

    if team2 in HOSTS and team1 not in HOSTS:
        return {
            "date": match["date"],
            "ts": date_to_ts(match["date"]),
            "homeSlug": team2,
            "awaySlug": team1,
            "homeName": match["team2"],
            "awayName": match["team1"],
            "hg": g2,
            "ag": g1,
            "neutral": False,
        }

    return {
        "date": match["date"],
        "ts": date_to_ts(match["date"]),
        "homeSlug": team1,
        "awaySlug": team2,
        "homeName": match["team1"],
        "awayName": match["team2"],
        "hg": g1,
        "ag": g2,
        "neutral": team1 not in HOSTS,
    }


def match_fingerprint(match):
    return "|".join([
        match.get("date", ""),
        match["homeSlug"],
        match["awaySlug"],
        str(match["hg"]),
        str(match["ag"]),
    ])


def load_json_matches(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    matches = []
    for raw in data.get("matches", []):
        if raw.get("status") != "FT":
            continue
        if raw.get("g1") is None or raw.get("g2") is None:
            continue
        matches.append(infer_world_cup_home_side(raw))

    matches.sort(key=lambda m: (m["ts"], m["homeSlug"], m["awaySlug"]))
    return matches


def cli_match(args):
    team1 = sanitize_slug(args.team1)
    team2 = sanitize_slug(args.team2)
    match = {
        "date": args.date,
        "ts": date_to_ts(args.date),
        "homeSlug": team1,
        "awaySlug": team2,
        "homeName": args.team1,
        "awayName": args.team2,
        "hg": args.g1,
        "ag": args.g2,
        "neutral": args.neutral or team1 not in HOSTS,
    }

    if args.venue_home == "team2":
        match = {
            **match,
            "homeSlug": team2,
            "awaySlug": team1,
            "homeName": args.team2,
            "awayName": args.team1,
            "hg": args.g2,
            "ag": args.g1,
            "neutral": args.neutral or team2 not in HOSTS,
        }
    elif args.venue_home == "neutral":
        match["neutral"] = True

    return match


def load_ratings(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ratings = {team: float(rating) for team, rating in data["ratings"].items()}
    return data, ratings


def apply_matches(ratings, matches, already_applied, force=False):
    touched = {}
    applied_ids = []
    skipped = 0

    for match in matches:
        fingerprint = match_fingerprint(match)
        if fingerprint in already_applied and not force:
            skipped += 1
            continue

        home = match["homeSlug"]
        away = match["awaySlug"]
        before_home = ratings.get(home, 1500.0)
        before_away = ratings.get(away, 1500.0)
        ratings.setdefault(home, before_home)
        ratings.setdefault(away, before_away)

        home_advantage = 0 if match["neutral"] else UPDATE_HOME_ADVANTAGE
        delta = rating_update(
            before_home,
            before_away,
            match["hg"],
            match["ag"],
            LEAGUE_NAME,
            ts=match["ts"],
            now_sec=match["ts"],
            home_advantage=home_advantage,
            use_recency=False,
        )

        ratings[home] = before_home + delta
        ratings[away] = before_away - delta
        touched.setdefault(home, before_home)
        touched.setdefault(away, before_away)
        applied_ids.append(fingerprint)

    return touched, applied_ids, skipped


def write_ratings(path, data, ratings, applied_ids, existing_ids):
    update_data = data.get("worldCupUpdate", {})
    all_ids = sorted(set(existing_ids) | set(applied_ids))
    update_data["matchesApplied"] = len(all_ids)
    update_data["lastUpdated"] = datetime.now(timezone.utc).isoformat()
    update_data["appliedMatchIds"] = all_ids

    data["matchesApplied"] = data.get("matchesApplied", 0) + len(applied_ids)
    data["worldCupUpdate"] = update_data
    data["ratings"] = {team: round(rating) for team, rating in sorted(ratings.items())}

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def print_changes(ratings, touched):
    if not touched:
        return
    print(f"\n{'team':<28} {'before':>7} {'after':>7} {'delta':>7}")
    for team in sorted(touched, key=lambda t: ratings[t] - touched[t], reverse=True):
        before = round(touched[team])
        after = round(ratings[team])
        print(f"  {team:<26} {before:>7} {after:>7} {after - before:>+7}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update calibrated Elo ratings with finished World Cup results."
    )
    parser.add_argument("--ratings", default=DEFAULT_RATINGS_FILE)
    parser.add_argument("--output", default=None)
    parser.add_argument("--json", default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--force", action="store_true",
                        help="Apply matches even if they were already recorded.")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--team1")
    parser.add_argument("--team2")
    parser.add_argument("--g1", type=int)
    parser.add_argument("--g2", type=int)
    parser.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--venue-home", choices=("team1", "team2", "neutral"), default="team1")
    parser.add_argument("--neutral", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or args.ratings
    single_match = any(v is not None for v in (args.team1, args.team2, args.g1, args.g2))
    if single_match and not all(v is not None for v in (args.team1, args.team2, args.g1, args.g2)):
        raise SystemExit("For a single match, provide --team1 --team2 --g1 --g2.")

    data, ratings = load_ratings(args.ratings)
    existing_ids = data.get("worldCupUpdate", {}).get("appliedMatchIds", [])
    matches = [cli_match(args)] if single_match else load_json_matches(args.json)

    touched, applied_ids, skipped = apply_matches(
        ratings,
        matches,
        set(existing_ids),
        force=args.force,
    )

    if not args.dry_run:
        write_ratings(output, data, ratings, applied_ids, existing_ids)

    action = "Would apply" if args.dry_run else "Applied"
    print(f"{action} {len(applied_ids)} World Cup matches; skipped {skipped}.")
    if not args.dry_run:
        print(f"Wrote {output}")
    print_changes(ratings, touched)


if __name__ == "__main__":
    main()
