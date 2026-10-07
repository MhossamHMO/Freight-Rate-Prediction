"""Diagnostics on the fitted model that back up claims made in the report."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import FreightRateModel


def quote_signal_correlations(dev: pd.DataFrame, market_index_by_day: pd.Series,
                              model: FreightRateModel) -> pd.DataFrame:
    """Within-month correlation of quote_signal with the rate error, January-October.

    'simple' is measured against a first-look baseline (distance, equipment, market index);
    'full' against the final model, which already models the calendar level.
    """
    keep = model.train_keep_
    y = np.log(dev["posted_rate"] / dev["distance"])
    full_resid = y - model.components(dev, market_index_by_day).sum(axis=1)

    X = np.column_stack([
        np.ones(len(dev)), np.log(dev["distance"]), np.log(dev["distance"]) ** 2,
        np.log(dev["market_index"].fillna(dev["date"].map(market_index_by_day))),
        dev["equipment"].eq("Reefer"), dev["equipment"].eq("Flatbed"),
    ]).astype(float)
    beta, *_ = np.linalg.lstsq(X[keep], y[keep].to_numpy(), rcond=None)
    simple_resid = y - X @ beta

    rows = []
    for month in range(1, 11):
        sel = ((dev["date"].dt.month == month) & keep).to_numpy()
        q = dev["quote_signal"].to_numpy()[sel]
        rows.append({"month": month,
                     "simple": np.corrcoef(q, simple_resid.to_numpy()[sel])[0, 1],
                     "full": np.corrcoef(q, full_resid.to_numpy()[sel])[0, 1]})
    return pd.DataFrame(rows).set_index("month")
