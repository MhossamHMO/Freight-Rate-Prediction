"""Validation harness: out-of-time folds, unseen-city folds, and shared metrics.

The final test set is the two months after the development data and contains
cities the development data never shows, so both of those situations are
reproduced here instead of relying on a random split.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .model import FreightRateModel

# (name, first test day, first day after the test window).  Training always uses
# every development row strictly before the first test day.
TEMPORAL_FOLDS = [
    ("test May-Jun", "2025-05-01", "2025-07-01"),
    ("test Jul-Aug", "2025-07-01", "2025-09-01"),
    ("test Aug-Sep", "2025-08-01", "2025-10-01"),
    ("test Sep-Oct", "2025-09-01", "2025-11-01"),
]


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    err = predicted - actual
    ape = np.abs(err) / actual
    return {
        "MAE ($)": float(np.abs(err).mean()),
        "RMSE ($)": float(np.sqrt((err ** 2).mean())),
        "MAPE (%)": float(100 * ape.mean()),
        "median APE (%)": float(100 * np.median(ape)),
        "log RMSE": float(np.sqrt((np.log(predicted / actual) ** 2).mean())),
    }


def corruption_flags(dev: pd.DataFrame, market_index_by_day: pd.Series) -> np.ndarray:
    """True for development rows whose label is corrupted.

    Taken from one reference fit on all development rows and then held fixed, so
    every candidate model is judged on the same 'clean' rows.
    """
    ref = FreightRateModel().fit(dev, market_index_by_day)
    return ~ref.train_keep_


def _evaluate(model, train, test, test_corrupt, mi) -> dict:
    model.fit(train, mi)
    pred = model.predict(test, mi)
    actual = test["posted_rate"].to_numpy()
    clean = score(actual[~test_corrupt], pred[~test_corrupt])
    every = score(actual, pred)
    return {**{f"clean {k}": v for k, v in clean.items()},
            "all-rows MAE ($)": every["MAE ($)"], "all-rows MAPE (%)": every["MAPE (%)"]}


def temporal_cv(dev: pd.DataFrame, mi: pd.Series, make_model: Callable, corrupt: np.ndarray,
                folds: Iterable = TEMPORAL_FOLDS) -> pd.DataFrame:
    rows = []
    for name, start, stop in folds:
        start, stop = pd.Timestamp(start), pd.Timestamp(stop)
        train = dev[dev["date"] < start]
        test_mask = ((dev["date"] >= start) & (dev["date"] < stop)).to_numpy()
        test = dev[test_mask]
        rows.append({"fold": name, "n_train": len(train), "n_test": len(test),
                     **_evaluate(make_model(), train, test, corrupt[test_mask], mi)})
    return pd.DataFrame(rows).set_index("fold")


def random_kfold_cv(dev: pd.DataFrame, mi: pd.Series, make_model: Callable, corrupt: np.ndarray,
                    k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Ordinary shuffled K-fold, kept only to show how optimistic it is for this problem."""
    order = np.random.default_rng(seed).permutation(len(dev))
    rows = []
    for i, fold_idx in enumerate(np.array_split(order, k)):
        test_mask = np.zeros(len(dev), dtype=bool)
        test_mask[fold_idx] = True
        rows.append({"fold": f"fold {i + 1}", "n_train": int((~test_mask).sum()), "n_test": len(fold_idx),
                     **_evaluate(make_model(), dev[~test_mask], dev[test_mask], corrupt[test_mask], mi)})
    return pd.DataFrame(rows).set_index("fold")


def unseen_city_cv(dev: pd.DataFrame, mi: pd.Series, make_model: Callable, corrupt: np.ndarray,
                   n_held_out: int = 8, repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Out-of-time AND out-of-city: hold a few cities out of training entirely.

    Test rows: September-October loads that touch at least one held-out city.
    'unseen' trains on pre-September rows that touch none of the held-out cities;
    'seen' trains on all pre-September rows (same test rows), as a control.
    """
    rng = np.random.default_rng(seed)
    cities = np.array(sorted(set(dev["pickup"]) | set(dev["delivery"])))
    cutoff = pd.Timestamp("2025-09-01")
    before = (dev["date"] < cutoff).to_numpy()
    rows = []
    for r in range(repeats):
        held = set(rng.choice(cities, n_held_out, replace=False))
        touches = (dev["pickup"].isin(held) | dev["delivery"].isin(held)).to_numpy()
        test_mask = ~before & touches
        test = dev[test_mask]
        for label, train in (("unseen", dev[before & ~touches]), ("seen", dev[before])):
            rows.append({"repeat": r, "cities": label, "n_train": len(train), "n_test": len(test),
                         **_evaluate(make_model(), train, test, corrupt[test_mask], mi)})
    return pd.DataFrame(rows).set_index(["cities", "repeat"]).sort_index()
