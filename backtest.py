import json
import math

import numpy as np

from time import time

from elo import match_prob, expected_score, rating_update

from config import (
    SEED,  MATCHES, 
    HOME_ADVANTAGE,
    OUTPUT_DIR
)
from calibrate import now_sec


UPDATE_HOME_ADV = HOME_ADVANTAGE / 2


BURN_IN = 350
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

def rps3(p, y):
    return 0.5 * ((p[0] - y[0]) ** 2 + (p[0] + p[1] - y[0] - y[1]) ** 2)


n = 0
hit = 0
brier, logloss = 0, 0
fav_N, fav_Hit = 0, 0
base_home, base_elo = 0, 0

eH, eD, eA = 0, 0, 0
rps, rps_u = 0, 0


if __name__ == "__main__":
    BINS = 10
    calib = [{"sumP": 0, "sumY": 0, "n": 0} for _ in range(BINS)]

    i = 0
    for match in MATCHES:
        if match['hg'] is None or match['ag'] is None:
            continue
        rating_a = getR(match['homeSlug'], match['homeName'])
        rating_b = getR(match['awaySlug'], match['awayName'])
        
        if i >= BURN_IN:
            probs = match_prob(rating_a, rating_b, HOME_ADVANTAGE)
            actual = 0 if match['hg'] > match['ag'] else 2 if match['hg'] < match['ag'] else 1
            y = [1.0 if actual == 0 else 0.0, 1.0 if actual == 1 else 0.0, 1.0 if actual == 2 else 0.0]
            pred = np.argmax(probs)
            if pred == actual:
                hit += 1

            brier += (probs[0] - y[0]) ** 2 + (probs[1] - y[1]) ** 2 + (probs[2] - y[2]) ** 2 
            logloss += -math.log(max(1e-15, probs[actual]))
            rps += rps3(probs, y)
            rps_u += rps3([1/3, 1/3, 1/3], y)
            for j in range(3):
                b = min(BINS - 1, math.floor(probs[j] * BINS))
                calib[b]["sumP"] += probs[j]
                calib[b]["sumY"] += y[j]
                calib[b]["n"] += 1
            
            if max(probs) > 0.5:
                fav_N += 1
                if pred == actual:
                    fav_Hit += 1
            if actual == 0:
                base_home += 1
            predicted = 0 if expected_score(rating_a, rating_b, HOME_ADVANTAGE) >= 0.5 else 2
            if predicted == actual:
                base_elo += 1
            if actual == 0:
                eH += 1
            elif actual == 1:
                eD += 1
            else:
                eA += 1
            n += 1

        delta = rating_update(
            rating_a, rating_b, match['hg'], match['ag'],
            match['leagueName'], ts=match['ts'], now_sec=now_sec(),
            home_advantage=UPDATE_HOME_ADV, use_recency=False
        )
        setR(match['homeSlug'], match['homeName'], rating_a + delta)
        setR(match['awaySlug'], match['awayName'], rating_b - delta)
        i += 1


    pct = lambda x: f"{x * 100:.1f}%"
    print(f"\n=== Walk-forward backtest — {n} of {len(MATCHES)} matches (burn-in {BURN_IN}) ===")
    print(f"Eval outcome split: home {pct(eH/n)}  draw {pct(eD/n)}  away {pct(eA/n)}\n")
    print(f"MODEL")
    print(f"  Accuracy (top pick):   {pct(hit/n)}")
    print(f"  Favourite acc (p≥50%): {pct(fav_Hit/fav_N)}  ({fav_N} matches)")
    print(f"  Brier (3-way, ↓):      {(brier/n):.3f}")
    print(f"  Log-loss (↓):          {(logloss/n):.3f}")
    print(f"  RPS (↓):               {(rps/n):.4f}")

    ece = sum(abs(b["sumP"] / b["n"] - b["sumY"] / b["n"]) * b["n"] if b["n"] else 0 for b in calib) / (3 * n)
    print(f"  ECE (calibration, ↓):  {(ece * 100):.1f}%\n")
    print(f"BASELINES (same matches)")
    print(f"  Always pick home:      {pct(base_home/n)}")
    print(f"  Pick higher-Elo team:  {pct(base_elo/n)}")
    print(f"  Coin-flip (uniform):   Brier {(2*(1/3)**2+(1-1/3)**2):.3f} · log-loss {(-math.log(1/3)):.3f} · RPS {(rps_u/n):.4f}\n")

    print(f"CALIBRATION (reliability — predicted vs observed per probability band)")

    for k, b in enumerate(calib):
        if not b["n"]:
            continue
        print(f"  {k*10}–{(k+1)*10}%   model said {(b['sumP']/b['n']*100):.0f}%  →  happened {(b['sumY']/b['n']*100):.0f}%   (n={b['n']})")


    with open(f"{OUTPUT_DIR}/model-backtest.json", "w", encoding="utf-8") as f:
        json.dump({
            "generatedAt": time(),
            "method": "Walk-forward out-of-sample: each match predicted from ratings built only on prior matches; Elo updated after. Burn-in skipped.",
            "totalMatches": len(MATCHES), "evaluated": n, "burnIn": BURN_IN,
            "outcomeSplit": {"home": round(eH/n, 4), "draw": round(eD/n, 4), "away": round(eA/n, 4)},
            "model": {"accuracy": round(hit/n, 4), "brier": round(brier/n, 4), "logloss": round(logloss/n, 4),
                    "rps": round(rps/n, 4), "ece": round(ece, 4), "favouriteAccuracy": round(fav_Hit/fav_N, 4), "favouriteCount": fav_N},
            "baselines": {"alwaysHome": round(base_home/n, 4), "eloPickNoDraw": round(base_elo/n, 4),
                        "uniformBrier": 0.6667, "uniformLogloss": 1.0986, "uniformRps": round(rps_u/n, 4)},
            "calibration": {"bins": [{"range":[k/10,(k+1)/10], "n":b["n"],
                                    "avgPred": round(b["sumP"]/b["n"], 4) if b["n"] else None,
                                    "obsFreq": round(b["sumY"]/b["n"], 4) if b["n"] else None} for k,b in enumerate(calib)],
                            "ece": round(ece, 4)},
        }, f, indent=4)
    print(f"→ wrote {OUTPUT_DIR}/model-backtest.json")
