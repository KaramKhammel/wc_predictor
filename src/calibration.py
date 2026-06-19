import json
import os
from time import time

from elo import rating_update
from config import (
    SEED,  MATCHES, 
    HOSTS, PARTICIPANTS,
    HOME_ADVANTAGE,
    CALIBRATED_RATINGS_FILE,
    CALIBRATION_START_DATE,
)

UPDATE_HOME_ADV = HOME_ADVANTAGE / 2
    

def now_sec():
    """Returns the current time in seconds since the epoch."""
    if MATCHES:
        return MATCHES[-1]["ts"]
    return int(time())


R = {}
def getR(slug, name):
    """Retrieves the Elo rating for a team based on its slug or name."""
    k = slug if slug is not None else f"ghost:{name}"
    if k not in R:
        R[k] = SEED.get(slug, 1500) if slug is not None else 1500
    return R[k]

def setR(slug, name, v):
    """Sets the Elo rating for a team based on its slug or name."""
    R[slug if slug is not None else f"ghost:{name}"] = v


def before_calibration_start(match):
    return (
        CALIBRATION_START_DATE is not None
        and match.get("date") is not None
        and match["date"] < CALIBRATION_START_DATE
    )


if __name__ == "__main__":
    applied = 0
    for match in MATCHES:
        if match['hg'] is None or match['ag'] is None:
            continue
        if before_calibration_start(match):
            continue
        rating_a = getR(match['homeSlug'], match['homeName'])
        rating_b = getR(match['awaySlug'], match['awayName'])

        # home_advantage = HOME_ADVANTAGE / 2 if match['homeSlug'] in HOSTS else 0
        # expected = expected_score(rating_a, rating_b, home_advantage)
        
        # score = 1.0 if match['hg'] > match['ag'] else 0.0 if match['hg'] < match['ag'] else 0.5
        # K = base_K(match['leagueName']) * recency(match['ts'], now_sec()) * g_mult(match['hg'] - match['ag'])
        # delta = K * (score - expected)
        delta = rating_update(
            rating_a, rating_b, match['hg'], match['ag'],
            match['leagueName'], ts=match['ts'], now_sec=now_sec(),
            home_advantage=0 if match.get("neutral") else UPDATE_HOME_ADV,
            use_recency=False
        )
        setR(match['homeSlug'], match['homeName'], rating_a + delta)
        setR(match['awaySlug'], match['awayName'], rating_b - delta)
        applied += 1


    output_slugs = sorted(set(PARTICIPANTS) | {
        slug for slug in R
        if not slug.startswith("ghost:")
    })
    ratings = {}
    for slug in output_slugs:
        # ratings[slug] = round(0.65 * R.get(slug, SEED[slug]) + 0.35 * SEED[slug])
        ratings[slug] = round(R.get(slug, SEED.get(slug, 1500)))
    os.makedirs(os.path.dirname(CALIBRATED_RATINGS_FILE), exist_ok=True)
    with open(CALIBRATED_RATINGS_FILE, "w", encoding="utf-8") as f:
        json.dump({"matchesApplied": applied, "ratings": ratings}, f, indent=4)
    
    print(f"Calibrated {applied} matches -> {CALIBRATED_RATINGS_FILE}")
