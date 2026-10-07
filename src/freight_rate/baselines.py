"""Reference models used to put the main model's validation numbers in context."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import ORIGIN, EQUIPMENT_CODE


class MedianRatePerMile:
    """Distance x the median rate-per-mile of the equipment type."""

    def fit(self, df: pd.DataFrame, market_index_by_day: pd.Series):
        rpm = df["posted_rate"] / df["distance"]
        self.median_ = rpm.groupby(df["equipment"]).median()
        return self

    def predict(self, df: pd.DataFrame, market_index_by_day: pd.Series) -> np.ndarray:
        return df["equipment"].map(self.median_).to_numpy() * df["distance"].to_numpy()


class KitchenSinkGBM:
    """The 'throw every column at a boosted tree' approach, with no special handling.

    Uses the raw columns (including quote_signal, the row-level market_index, raw
    signed weight and a numeric day counter), trains on every row including the
    corrupted labels, and has no separate time component.
    """

    COLUMNS = ["distance", "equip", "weight", "market_index", "quote_signal",
               "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "day", "month"]

    def _table(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[["distance", "weight", "market_index", "quote_signal",
                "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].copy()
        X["equip"] = df["equipment"].map(EQUIPMENT_CODE)
        X["day"] = (df["date"] - ORIGIN).dt.days
        X["month"] = df["date"].dt.month
        return X[self.COLUMNS]

    def fit(self, df: pd.DataFrame, market_index_by_day: pd.Series):
        y = np.log(df["posted_rate"] / df["distance"])
        self.gbm_ = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, random_state=0,
            categorical_features=[self.COLUMNS.index("equip")],
        ).fit(self._table(df), y)
        return self

    def predict(self, df: pd.DataFrame, market_index_by_day: pd.Series) -> np.ndarray:
        return np.exp(self.gbm_.predict(self._table(df))) * df["distance"].to_numpy()
