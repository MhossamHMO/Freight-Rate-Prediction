"""Run every validation experiment and save the tables used in the report.

    python scripts/run_validation.py

Outputs (CSV) go to outputs/validation/.  Takes a few minutes.
"""
from __future__ import annotations

import functools
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import pandas as pd

from freight_rate.baselines import KitchenSinkGBM, MedianRatePerMile
from freight_rate.data import daily_market_index, load_development, load_validation
from freight_rate.model import FreightRateModel
from freight_rate.validation import (corruption_flags, random_kfold_cv, temporal_cv,
                                     unseen_city_cv)

OUT = ROOT / "outputs" / "validation"

# Six expanding-window folds: train on everything before the window, test on the
# two months after it.  Same shape as the real task (train Jan-Oct, predict Nov-Dec).
FOLDS = [("Apr-May", "2025-04-01", "2025-06-01"), ("May-Jun", "2025-05-01", "2025-07-01"),
         ("Jun-Jul", "2025-06-01", "2025-08-01"), ("Jul-Aug", "2025-07-01", "2025-09-01"),
         ("Aug-Sep", "2025-08-01", "2025-10-01"), ("Sep-Oct", "2025-09-01", "2025-11-01")]

CANDIDATES = {
    "Median rate-per-mile by equipment": MedianRatePerMile,
    "Boosted trees on all raw columns": KitchenSinkGBM,
    "Final model without geography": functools.partial(FreightRateModel, use_geography=False),
    "Final model without lane correction": functools.partial(FreightRateModel, lane_ridge=0.0),
    "Final model + quote_signal": functools.partial(FreightRateModel, use_quote_signal=True),
    "Final model": FreightRateModel,
}
HEADLINE = ["clean log RMSE", "clean MAPE (%)", "clean MAE ($)", "clean RMSE ($)", "all-rows MAPE (%)",
            "all-rows MAE ($)"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev, val = load_development(), load_validation()
    market = daily_market_index(dev, val)
    corrupt = corruption_flags(dev, market)
    print(f"{corrupt.sum()} development rows flagged as corrupted labels "
          f"({100 * corrupt.mean():.2f}%) - excluded from the 'clean' metrics")

    per_fold, summary = {}, {}
    for name, make in CANDIDATES.items():
        t0 = time.time()
        table = temporal_cv(dev, market, make, corrupt, FOLDS)
        per_fold[name] = table["clean log RMSE"]
        summary[name] = table[HEADLINE].mean()
        print(f"{name:40s} out-of-time log RMSE {table['clean log RMSE'].mean():.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if name == "Final model":
            table.to_csv(OUT / "temporal_folds_final.csv")
    pd.DataFrame(per_fold).to_csv(OUT / "temporal_log_rmse_by_fold.csv")
    pd.DataFrame(summary).T.to_csv(OUT / "temporal_summary.csv")

    # Why not a random split?  Same models, shuffled 5-fold vs the out-of-time folds above.
    rows = {}
    for name in ("Boosted trees on all raw columns", "Final model"):
        random_cv = random_kfold_cv(dev, market, CANDIDATES[name], corrupt)
        rows[name] = {"random 5-fold log RMSE": random_cv["clean log RMSE"].mean(),
                      "out-of-time log RMSE": summary[name]["clean log RMSE"],
                      "random 5-fold MAPE (%)": random_cv["clean MAPE (%)"].mean(),
                      "out-of-time MAPE (%)": summary[name]["clean MAPE (%)"]}
    pd.DataFrame(rows).T.to_csv(OUT / "random_vs_temporal.csv")
    print(pd.DataFrame(rows).T.round(4).to_string())

    # Unseen cities: 8 of the 64 cities removed from training, tested on Sep-Oct loads that touch them.
    cities = {}
    for name in ("Final model", "Final model without geography"):
        r = unseen_city_cv(dev, market, CANDIDATES[name], corrupt)
        cities[name] = r.groupby(level=0)[["clean log RMSE", "clean MAPE (%)", "clean MAE ($)"]].mean().stack()
    pd.DataFrame(cities).to_csv(OUT / "unseen_cities.csv")
    print(pd.DataFrame(cities).round(4).to_string())


if __name__ == "__main__":
    main()
