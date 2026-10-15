import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.explain import importance_table, permutation_table


def test_importance_table_sorts_by_shap():
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    shap_values = np.array([[1.0, -5.0], [-1.0, 5.0]])
    perm = pd.Series({"a": 2.0, "b": 9.0})
    t = importance_table(shap_values, X, perm)
    assert list(t.index) == ["b", "a"]
    assert t.loc["b", "mean_abs_shap_mwh"] == 5.0
    assert t.loc["a", "permutation_mae_increase_mwh"] == 2.0


def test_permutation_importance_finds_the_real_driver():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"signal": rng.normal(size=600), "noise": rng.normal(size=600)})
    y = 50 * X["signal"] + rng.normal(scale=1.0, size=600)
    model = HistGradientBoostingRegressor(random_state=0).fit(X, y)
    p = permutation_table(model, X, y, n_repeats=3)
    assert p["signal"] > 10 * max(p["noise"], 0.01)
