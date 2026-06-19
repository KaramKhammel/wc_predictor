import json
from datetime import datetime, timezone

from elo import rating_update
from config import (
    HOSTS,
    HOME_ADVANTAGE,
    DATA_DIR,
    OUTPUT_DIR,
    sanitize_slug,
)


UPDATE_HOME_ADV = HOME_ADVANTAGE / 2
LEAGUE_NAME = "World Cup"


def date_to_ts(date_str):
    return int(
        datetime.fromisoformat(date_str)
        .replace(hour=12, tzinfo=timezone.utc)
        .timestamp()
    )


def to_canonical(wc_match):
    """Convert a wc2026-results entry to the engine's match shape.

    Home side is the host country if one is playing, otherwise team1 with
    home_advantage forced to 0 (neutral venue)."""
    t1 = sanitize_slug(wc_match["t1"])
    t2 = sanitize_slug(wc_match["t2"])

    if t2 in HOSTS and t1 not in HOSTS:
        return {
            "ts": date_to_ts(wc_match["date"]),
            "date": wc_match["date"],
            "homeSlug": t2,
            "awaySlug": t1,
            "homeName": wc_match["team2"],
            "awayName": wc_match["team1"],
            "hg": wc_match["g2"],
            "ag": wc_match["g1"],
            "leagueName": LEAGUE_NAME,
            "neutral": False,
        }
    return {
        "ts": date_to_ts(wc_match["date"]),
        "date": wc_match["date"],
        "homeSlug": t1,
        "awaySlug": t2,
        "homeName": wc_match["team1"],
        "awayName": wc_match["team2"],
        "hg": wc_match["g1"],
        "ag": wc_match["g2"],
        "leagueName": LEAGUE_NAME,
        "neutral": t1 not in HOSTS,
    }


if __name__ == "__main__":
    with open(f"{OUTPUT_DIR}/elo-calibrated.json", "r", encoding="utf-8") as f:
        calibrated = json.load(f)

    R = {k: float(v) for k, v in calibrated["ratings"].items()}
    before = dict(R)

    with open(f"{DATA_DIR}/wc2026-results.json", "r", encoding="utf-8") as f:
        wc = json.load(f)

    matches = [
        to_canonical(m) for m in wc["matches"]
        if m.get("status") == "FT" and m.get("g1") is not None and m.get("g2") is not None
    ]
    matches.sort(key=lambda m: m["ts"])

    applied = 0
    for m in matches:
        ra = R.get(m["homeSlug"], 1500.0)
        rb = R.get(m["awaySlug"], 1500.0)
        home_adv = 0 if m["neutral"] else UPDATE_HOME_ADV

        delta = rating_update(
            ra, rb, m["hg"], m["ag"], m["leagueName"],
            ts=m["ts"], now_sec=m["ts"],
            home_advantage=home_adv, use_recency=False,
        )
        R[m["homeSlug"]] = ra + delta
        R[m["awaySlug"]] = rb - delta
        applied += 1

    ratings = {k: round(v) for k, v in R.items()}
    out = {
        "matchesApplied": calibrated.get("matchesApplied", 0) + applied,
        "wcMatchesApplied": applied,
        "ratings": ratings,
    }
    with open(f"{OUTPUT_DIR}/elo-calibrated.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    touched = sorted(
        {s for m in matches for s in (m["homeSlug"], m["awaySlug"])},
        key=lambda s: round(R[s]) - round(before.get(s, 1500.0)),
        reverse=True,
    )
    print(f"Applied {applied} WC matches -> {OUTPUT_DIR}/elo-calibrated.json\n")
    print(f"{'team':<28} {'before':>7} {'after':>7} {'delta':>7}")
    for slug in touched:
        b = round(before.get(slug, 1500.0))
        a = round(R[slug])
        print(f"  {slug:<26} {b:>7} {a:>7} {a - b:>+7}")
