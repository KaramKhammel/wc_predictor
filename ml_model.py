"""
The learned match model: a 1X2 outcome classifier + two Poisson goal regressors,
producing the SAME outputs as the Elo predictor (1X2 probabilities, expected goals,
a full exact-score matrix, and a most-likely scoreline) plus a small blend with the
Elo engine for the final probabilities.
"""
import numpy as np
from scipy.stats import poisson
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

from ml_config import (
    BLEND_W,
    CLF_PARAMS,
    GOAL_REG_PARAMS,
    MAX_GOALS,
)


class MatchModel:
    """Bundle of (classifier, home-goal regressor, away-goal regressor)."""

    def __init__(self, feature_names=None, blend_w=BLEND_W, max_goals=MAX_GOALS):
        self.clf = xgb.XGBClassifier(**CLF_PARAMS)
        self.reg_home = HistGradientBoostingRegressor(**GOAL_REG_PARAMS)
        self.reg_away = HistGradientBoostingRegressor(**GOAL_REG_PARAMS)
        self.feature_names = feature_names
        self.blend_w = blend_w
        self.max_goals = max_goals

    # ---- training -----------------------------------------------------------
    def fit(self, X, y_out, y_hg, y_ag):
        self.feature_names = list(X.columns)
        self.clf.fit(X, y_out)
        self.reg_home.fit(X, y_hg)
        self.reg_away.fit(X, y_ag)
        return self

    # ---- raw heads ----------------------------------------------------------
    def _align(self, X):
        if self.feature_names is not None:
            return X[self.feature_names]
        return X

    def predict_proba_clf(self, X):
        return self.clf.predict_proba(self._align(X))

    def predict_goals(self, X):
        Xa = self._align(X)
        lam_h = np.clip(self.reg_home.predict(Xa), 0.05, 8.0)
        lam_a = np.clip(self.reg_away.predict(Xa), 0.05, 8.0)
        return lam_h, lam_a

    # ---- combined outputs ----------------------------------------------------
    def predict_1x2(self, X):
        """Blended 1X2 probabilities: BLEND_W * ML classifier + (1-w) * Elo engine.

        The Elo engine probabilities are read directly from the feature columns
        (p_home / p_draw / p_away), so no separate engine call is needed.
        """
        ml = self.predict_proba_clf(X)
        Xa = self._align(X)
        eng = Xa[["p_home", "p_draw", "p_away"]].to_numpy()
        eng = eng / eng.sum(axis=1, keepdims=True)
        blend = self.blend_w * ml + (1.0 - self.blend_w) * eng
        return blend / blend.sum(axis=1, keepdims=True)

    def score_matrix(self, lam_h, lam_a):
        """Independent-Poisson exact-score matrix from predicted goal rates."""
        g = np.arange(self.max_goals + 1)
        ph = poisson.pmf(g, lam_h)
        pa = poisson.pmf(g, lam_a)
        M = np.outer(ph, pa)
        M /= M.sum()
        return M

    @staticmethod
    def matrix_to_1x2(M):
        idx = np.arange(M.shape[0])
        p_home = M[np.greater.outer(idx, idx)].sum()
        p_draw = np.trace(M)
        p_away = 1.0 - p_home - p_draw
        return float(p_home), float(p_draw), float(p_away)

    # ---- persistence ---------------------------------------------------------
    def save(self, path):
        import os
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)


# ---- metric helpers (shared by train_model.py) ------------------------------
def brier3(proba, y):
    Y = np.eye(3)[np.asarray(y)]
    return float(np.mean(np.sum((proba - Y) ** 2, axis=1)))


def rps3(proba, y):
    Y = np.eye(3)[np.asarray(y)]
    cp = np.cumsum(proba, axis=1)
    cy = np.cumsum(Y, axis=1)
    return float(np.mean(np.sum((cp - cy) ** 2, axis=1)) / (3 - 1))
