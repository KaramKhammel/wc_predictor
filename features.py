"""
Walk-forward, leakage-free feature engineering for the ML match model.

Every feature for match i is computed using ONLY matches < i, exactly the same
discipline used in backtesting.py: we read the engine state, emit features, record
the target, and ONLY THEN update the Elo state and rolling histories.

The Elo engine (elo.py) is reused as a feature generator so the learned model is
fed the engine's own opinions (win probabilities, expected goals, win expectancy)
alongside data-driven form / goal-difference / xG-conversion features.
"""
from collections import defaultdict, deque

from elo import (
    base_K,
    expected_goals,
    expected_score,
    match_prob,
    rating_update,
)
from config import HOME_ADVANTAGE, MATCHES
from ml_config import BURN_IN_MATCHES, FORM_WINDOWS

UPDATE_HOME_ADV = HOME_ADVANTAGE / 2

# outcome label encoding (matches predict ordering: home / draw / away)
OUTCOME_HOME, OUTCOME_DRAW, OUTCOME_AWAY = 0, 1, 2


def _load_seed():
    """Reuse the calibrated Elo seed so feature ratings match the engine."""
    from config import SEED
    return {k: float(v) for k, v in SEED.items()}


def _rolling_stats(history, window):
    """Form features for one team over its last `window` matches (past only).

    history entries: dict(result, gf, ga, elo_exp_goals)
      result: 1.0 win / 0.5 draw / 0.0 loss
    """
    dq = list(history)[-window:]
    n = len(dq)
    if n == 0:
        return dict(form=0.5, winrate=0.5, gd=0.0, gf=0.0, ga=0.0, xg_conv=1.0, n=0)
    pts = sum(e["result"] for e in dq)
    wins = sum(1 for e in dq if e["result"] == 1.0)
    gf = sum(e["gf"] for e in dq)
    ga = sum(e["ga"] for e in dq)
    exp = sum(e["elo_exp_goals"] for e in dq)
    return dict(
        form=pts / n,                       # avg points per match (win rate weighted)
        winrate=wins / n,
        gd=(gf - ga) / n,                   # avg goal difference
        gf=gf / n,
        ga=ga / n,
        xg_conv=(gf / exp) if exp > 0 else 1.0,  # actual goals / elo-expected goals
        n=n,
    )


def build_match_features(rating_home, rating_away, home_advantage,
                         home_history, away_history, league_name):
    """Build the feature dict for a single (future) match from current engine state.

    This is the SAME routine used both at training time (per historical match) and
    at prediction time (predict_ml.py), guaranteeing identical feature semantics.

    `home_history` / `away_history` are iterables of past-match dicts for each team.
    """
    ra, rb, ha = rating_home, rating_away, home_advantage

    p_home, p_draw, p_away = match_prob(ra, rb, ha)
    exp_g_home = expected_goals(ra, rb, ha)
    exp_g_away = expected_goals(rb, ra, -ha / 2)

    feat = {
        "elo_home": ra,
        "elo_away": rb,
        "elo_diff": ra - rb,
        "home_advantage": ha,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "exp_g_home": exp_g_home,
        "exp_g_away": exp_g_away,
        "exp_g_diff": exp_g_home - exp_g_away,
        "elo_exp_score": expected_score(ra, rb, ha),
        "importance": base_K(league_name),     # tournament weight / match importance
    }
    for w in FORM_WINDOWS:
        sh = _rolling_stats(home_history, w)
        sa = _rolling_stats(away_history, w)
        feat[f"home_form{w}"] = sh["form"]
        feat[f"away_form{w}"] = sa["form"]
        feat[f"home_winrate{w}"] = sh["winrate"]
        feat[f"away_winrate{w}"] = sa["winrate"]
        feat[f"home_gd{w}"] = sh["gd"]
        feat[f"away_gd{w}"] = sa["gd"]
        feat[f"home_gf{w}"] = sh["gf"]
        feat[f"away_gf{w}"] = sa["gf"]
        feat[f"home_ga{w}"] = sh["ga"]
        feat[f"away_ga{w}"] = sa["ga"]
        feat[f"home_xgconv{w}"] = sh["xg_conv"]
        feat[f"away_xgconv{w}"] = sa["xg_conv"]
        feat[f"form_diff{w}"] = sh["form"] - sa["form"]
        feat[f"gd_diff{w}"] = sh["gd"] - sa["gd"]
        ## add xg_conv_delta features for home and away teams
    return feat


def outcome_label(hg, ag):
    return OUTCOME_HOME if hg > ag else (OUTCOME_DRAW if hg == ag else OUTCOME_AWAY)


def build_training_table(matches=None, burn_in=BURN_IN_MATCHES, use_recency=False):
    """Walk forward through history and emit (features, targets, meta) leakage-free.

    Returns
    -------
    rows    : list[dict]   feature dicts
    y_out   : list[int]    0=home win, 1=draw, 2=away win
    y_hg    : list[int]    home goals (score target)
    y_ag    : list[int]    away goals (score target)
    meta    : list[dict]   {date, league, ts} per row
    final_ratings  : dict  Elo state after the full pass (latest ratings)
    final_history  : dict  team -> deque of recent matches after the full pass
    """
    if matches is None:
        matches = MATCHES

    seed = _load_seed()
    ratings = {}
    histories = defaultdict(lambda: deque(maxlen=max(FORM_WINDOWS)))

    def get_rating(slug):
        if slug not in ratings:
            ratings[slug] = seed.get(slug, 1500.0)
        return ratings[slug]

    rows, y_out, y_hg, y_ag, meta = [], [], [], [], []

    usable = [m for m in matches if m.get("hg") is not None and m.get("ag") is not None]
    for i, m in enumerate(usable):
        hs, as_ = m["homeSlug"], m["awaySlug"]
        ra, rb = get_rating(hs), get_rating(as_)
        neutral = m.get("neutral", False)
        ha = 0 if neutral else HOME_ADVANTAGE

        feat = build_match_features(
            ra, rb, ha, histories[hs], histories[as_], m["leagueName"]
        )
        hg, ag = m["hg"], m["ag"]

        if i >= burn_in:
            rows.append(feat)
            y_out.append(outcome_label(hg, ag))
            y_hg.append(hg)
            y_ag.append(ag)
            meta.append({"date": m["date"], "league": m["leagueName"], "ts": m["ts"]})

        # ---- update engine state AFTER recording (no leakage) ----
        exp_g_home = feat["exp_g_home"]
        exp_g_away = feat["exp_g_away"]
        delta = rating_update(
            ra, rb, hg, ag, m["leagueName"], ts=m["ts"],
            home_advantage=(0 if neutral else UPDATE_HOME_ADV),
            use_recency=use_recency,
        )
        ratings[hs] = ra + delta
        ratings[as_] = rb - delta

        res = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        histories[hs].append(dict(result=res, gf=hg, ga=ag, elo_exp_goals=exp_g_home))
        histories[as_].append(
            dict(result=(0.5 if res == 0.5 else 1.0 - res), gf=ag, ga=hg,
                 elo_exp_goals=exp_g_away)
        )

    return rows, y_out, y_hg, y_ag, meta, dict(ratings), dict(histories)
