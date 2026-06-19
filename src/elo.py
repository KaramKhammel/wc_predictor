from math import exp, factorial
from random import Random, random

_RNG = Random()

# This module implements the Elo rating system for predicting match outcomes in the World Cup.


# The K-factor is a parameter that controls the sensitivity of the Elo rating updates to match outcomes.
# A higher K-factor means that the ratings will change more significantly after each match, 
# while a lower K-factor means that the ratings will change less.
K_FACTOR_WC = 55

# The DC Rho value is a parameter that controls the influence of the DC Tau value on the Elo rating updates. 
# A negative value means that the DC Tau value will have a negative influence on the rating updates, 
# while a positive value means that it will have a positive influence. 
# The optimal value for DC Rho may vary depending on the specific use case and dataset, 
# and it may require tuning to achieve the best performance.
DC_RHO = -0.13 


def dcTau(a, b, lambda_, mu, rho):
    """Returns the DC Tau value for a given match outcome and parameters."""
    if a == 0 and b == 0:
        return 1 - lambda_ * mu * rho
    if a == 0 and b == 1:
        return 1 + lambda_ * mu * rho
    if a == 1 and b == 0:
        return 1 + mu * rho
    if a == 1 and b == 1:
        return 1 - rho
    return 1


# Elo win expectancy (logistic on rating difference)
def expected_score(rating_a, rating_b, home_advantage=0):
    """Returns the expected score for team A against team B based on their Elo ratings."""
    return 1 / (1 + 10 ** ((rating_b - rating_a + home_advantage) / 400))


# Rating difference to expected goals (Poisson lambda)
# Flat denominator keeps single-match variance near real football upset frequency
def expected_goals(rating_a, rating_b, home_advantage=0):
    """Returns the expected number of goals for team A against team B based on their Elo ratings."""
    diff = rating_a - rating_b + home_advantage
    lambda_ = 1.35 + diff / 400
    return max(0.3, min(3.5, lambda_))


def poisson_pmf(k, lambda_):
    if lambda_ <= 0:
        if k == 0:
            return 1.0
        return 0.0
    return (lambda_ ** k) * exp(-lambda_) / factorial(k)


def poisson_sample(lambda_):
    L = exp(-lambda_)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random()
    return k - 1


def match_prob(rating_a, rating_b, home_advantage=0):
    """Returns the probabilities of a win, draw, and loss for team A against team B."""
    
    lambda_a = expected_goals(rating_a, rating_b, home_advantage)
    lambda_b = expected_goals(rating_b, rating_a, -home_advantage/2)

    pA = 0.0
    pD = 0.0
    pB = 0.0

    for goals_a in range(8):
        pmf_a = poisson_pmf(goals_a, lambda_a)
        for goals_b in range(8):
            tau = dcTau(goals_a, goals_b, lambda_a, lambda_b, DC_RHO)
            p = pmf_a * poisson_pmf(goals_b, lambda_b) * tau
            if goals_a > goals_b:
                pA += p
            elif goals_a < goals_b:
                pB += p
            else:
                pD += p

    total = pA + pD + pB

    return pA/total , pD/ total, pB/total


def sample_match(rating_a, rating_b, home_advantage=0):
    """Simulates a match between team A and team B based on their Elo ratings."""
    lambda_a = expected_goals(rating_a, rating_b, home_advantage)
    lambda_b = expected_goals(rating_b, rating_a, -home_advantage/2)

    goals_a = poisson_sample(lambda_a)
    goals_b = poisson_sample(lambda_b)

    if goals_a == goals_b:
        # In the case of a draw, we can use the expected score to determine which team is more likely to have scored the goals.
        if random() < expected_score(rating_a, rating_b, home_advantage):
            goals_a += 1
        else:
            goals_b += 1

    return goals_a, goals_b


def base_K(league_name):
    name = league_name.lower()

    if "world cup" in name and "qual" not in name:
        return K_FACTOR_WC
    if "world cup" in name and ("qual" in name or "qualification" in name):
        return 42
    if any(x in name for x in ["copa america", "euro championship", "asian cup", "africa cup", "gold cup"]):
        return 50
    if any(x in name for x in ["nations league", "nations cup"]):
        return 32
    if "friendl" in name:
        return 18
    return 28


def recency(ts_sec, now_sec):
    """Calculates the recency factor for a match based on its timestamp."""
    return 0.5 ** (((now_sec - ts_sec) / (30.44 * 86400)) / 18)  # 18-mo half-life

def g_mult(gd):
    """Calculates the goal difference multiplier for a match."""
    d = abs(gd)
    if d <= 1:
        return 1
    elif d == 2:
        return 1.5
    else:
        return (11 + d) / 8


def rating_update(rating_a, rating_b, hg, ag, league_name, ts=None, now_sec=None,
                  home_advantage=0, use_recency=False):
    expected = expected_score(rating_a, rating_b, home_advantage)
    score = 1.0 if hg > ag else 0.0 if hg < ag else 0.5
    K = base_K(league_name) * g_mult(hg - ag)
    if use_recency and ts is not None and now_sec is not None:
        K *= recency(ts, now_sec)
    return K * (score - expected)


MAX_GOALS = 8  # maximum number of goals to consider in the scoreline probability matrix

def score_matrix(rating_a, rating_b, home_advantage=0, max_goals=MAX_GOALS):
    """Return (matrix, lambda_a, lambda_b) where matrix[i][j] is the normalized
    Dixon-Coles probability of the exact scoreline i-j. Sums to 1."""
    lambda_a = expected_goals(rating_a, rating_b, home_advantage)
    lambda_b = expected_goals(rating_b, rating_a, -home_advantage / 2)

    matrix = [[0.0] * (max_goals + 1) for _ in range(max_goals + 1)]
    total = 0.0
    for ga in range(max_goals + 1):
        pmf_a = poisson_pmf(ga, lambda_a)
        for gb in range(max_goals + 1):
            tau = dcTau(ga, gb, lambda_a, lambda_b, DC_RHO)
            p = pmf_a * poisson_pmf(gb, lambda_b) * tau
            matrix[ga][gb] = p
            total += p
    if total > 0:
        for ga in range(max_goals + 1):
            for gb in range(max_goals + 1):
                matrix[ga][gb] /= total
    return matrix, lambda_a, lambda_b


def simulate_match(rating_a, rating_b, home_advantage=0, knockout=False, rng=None):
    """Monte Carlo sample of a single match, consistent with match_prob().

    Draws an exact scoreline from the Dixon-Coles joint distribution.
    - knockout=False (group stage): returns the real scoreline; draws are kept.
    - knockout=True: if the drawn scoreline is a tie, resolve it like a
      penalty shootout using the Elo win expectancy (no fake extra goal added
      to the scoreline -- we return the regulation score plus a winner flag).

    Args:
        rng: optional random.Random instance for independent/parallel streams.
             Defaults to the seedable module RNG.

    Returns:
        dict with keys: goals_a, goals_b, outcome ('A'|'D'|'B'),
        winner ('A'|'B'|None). For knockout draws, winner is resolved but
        outcome stays 'D' (the regulation result).
    """
    r = rng if rng is not None else _RNG
    matrix, _, _ = score_matrix(rating_a, rating_b, home_advantage)
    max_goals = len(matrix) - 1

    # Inverse-CDF sample over the flattened scoreline grid.
    x = r.random()
    cum = 0.0
    goals_a, goals_b = 0, 0
    found = False
    for ga in range(max_goals + 1):
        for gb in range(max_goals + 1):
            cum += matrix[ga][gb]
            if x <= cum:
                goals_a, goals_b = ga, gb
                found = True
                break
        if found:
            break

    if goals_a > goals_b:
        outcome, winner = "A", "A"
    elif goals_a < goals_b:
        outcome, winner = "B", "B"
    else:
        outcome = "D"
        winner = None
        if knockout:
            # Resolve the tie (ET/penalties) via Elo win expectancy.
            winner = "A" if r.random() < expected_score(rating_a, rating_b, home_advantage) else "B"

    return {"goals_a": goals_a, "goals_b": goals_b, "outcome": outcome, "winner": winner}


def predicted_scoreline(rating_a, rating_b, home_advantage=0):
    """Most likely exact scoreline, consistent with the predicted outcome."""
    pA, pD, pB = match_prob(rating_a, rating_b, home_advantage)
    matrix, _, _ = score_matrix(rating_a, rating_b, home_advantage)
    n = len(matrix)

    if pA >= pD and pA >= pB:
        keep = lambda i, j: i > j
    elif pB >= pA and pB >= pD:
        keep = lambda i, j: i < j
    else:
        keep = lambda i, j: i == j

    best_ga, best_gb, best_p = 0, 0, -1.0
    for i in range(n):
        for j in range(n):
            if keep(i, j) and matrix[i][j] > best_p:
                best_ga, best_gb, best_p = i, j, matrix[i][j]
    return best_ga, best_gb, best_p

