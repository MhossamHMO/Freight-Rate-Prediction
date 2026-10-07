"""Two-part model for log(rate per mile).

    log(rate / distance) = G(static load features) + T(date, market, equipment)

* G is a gradient-boosted tree model.  It only sees things that do not move with
  time: distance, equipment, weight and geography.  Because nothing in G depends
  on the calendar it cannot memorise a month, which is what makes it safe to apply
  to months it never saw.
* T is a small linear model: a drift per day, a quarter-end ramp whose size
  depends on equipment, and the (log) market index with a distance-dependent
  elasticity.  Being linear it extrapolates sensibly into November/December
  instead of freezing at the last value seen, as a tree would.

The two parts are fitted by back-fitting.  Rows whose label is badly corrupted
(rate off by a factor of 2x or more) are detected from the fit and left out of
training; they are not predictable from the features.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from .data import EQUIPMENT, build_features

STATIC_FEATURES = [
    "log_distance", "equipment", "weight",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "lat_mid", "lon_mid", "d_lat", "d_lon",
]
CATEGORICAL = [STATIC_FEATURES.index("equipment")]

DEFAULT_GBM = dict(
    max_iter=500, learning_rate=0.05, max_leaf_nodes=24,
    min_samples_leaf=30, l2_regularization=1.0,
)


def time_design(X: pd.DataFrame, mean_log_distance: float) -> np.ndarray:
    """Design matrix of the linear market/time part T."""
    cols = [np.ones(len(X)), X["day"].to_numpy() / 100.0]
    for code in range(len(EQUIPMENT)):
        cols.append(X["quarter_end"].to_numpy() * (X["equipment"].to_numpy() == code))
    log_market = X["log_market"].to_numpy()
    cols.append(log_market)
    cols.append(log_market * (X["log_distance"].to_numpy() - mean_log_distance))
    return np.column_stack(cols)


TIME_TERMS = ["intercept", "drift_per_100d", *[f"quarter_end_ramp[{e}]" for e in EQUIPMENT],
              "market_elasticity", "market_elasticity_x_log_distance"]


@dataclass
class FreightRateModel:
    gbm_params: dict = field(default_factory=lambda: dict(DEFAULT_GBM))
    backfit_rounds: int = 4
    corruption_cutoff: float = 0.25        # |log error| beyond this is not noise
    lane_ridge: float = 3.0                # shrinkage of the city/lane correction (0 = off)
    use_geography: bool = True             # switches below exist for the ablations only
    use_quote_signal: bool = False

    # ---- fitting -----------------------------------------------------------------
    def _static(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = STATIC_FEATURES if self.use_geography else STATIC_FEATURES[:3]
        if self.use_quote_signal:
            cols = [*cols, "quote_signal"]
        return X[cols].assign(weight=X["weight"].fillna(self.weight_fill_))

    def _new_gbm(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            categorical_features=CATEGORICAL, random_state=0, **self.gbm_params
        )

    def _backfit(self, X, y, keep) -> None:
        design = time_design(X, self.mean_log_distance_)
        T = np.zeros(len(X))
        for _ in range(self.backfit_rounds):
            self.gbm_ = self._new_gbm().fit(self._static(X)[keep], (y - T)[keep])
            e = y - self.gbm_.predict(self._static(X))
            self.time_coef_, *_ = np.linalg.lstsq(design[keep], e[keep], rcond=None)
            T = design @ self.time_coef_

    def fit(self, df: pd.DataFrame, market_index_by_day: pd.Series) -> "FreightRateModel":
        X = build_features(df, market_index_by_day)
        y = np.log(df["posted_rate"].to_numpy() / df["distance"].to_numpy())
        self.mean_log_distance_ = float(X["log_distance"].mean())
        self.weight_fill_ = float(X["weight"].median())     # for the ~1% of loads with no weight

        keep = np.ones(len(X), dtype=bool)
        for _ in range(2):                   # fit, flag corrupted labels, refit
            self._backfit(X, y, keep)
            resid = y - self._raw_log_rpm(X)
            keep = np.abs(resid) < self.corruption_cutoff
        if self.lane_ridge > 0:
            self._fit_lane_correction(df, resid, keep)
            resid = y - self._raw_log_rpm(X) - self._lane_term(df)
        self.train_keep_ = keep
        self.n_flagged_ = int((~keep).sum())
        self.residual_sd_ = float(resid[keep].std())
        return self

    # ---- shrunk correction for cities / lanes seen in training ----------------------
    # Pickup city, delivery city and directed lane each get a ridge-shrunk offset on
    # top of G.  Cities or lanes never seen in training simply get no correction.
    @staticmethod
    def _lane_codes(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"pickup": df["pickup"], "delivery": df["delivery"],
                             "lane": df["pickup"] + ">" + df["delivery"]})

    def _fit_lane_correction(self, df, resid, keep) -> None:
        codes = self._lane_codes(df)
        self.lane_encoder_ = OneHotEncoder(handle_unknown="ignore").fit(codes[keep])
        self.lane_ridge_ = Ridge(alpha=self.lane_ridge).fit(
            self.lane_encoder_.transform(codes[keep]), resid[keep])

    def _lane_term(self, df: pd.DataFrame) -> np.ndarray:
        if self.lane_ridge <= 0:
            return 0.0
        return self.lane_ridge_.predict(self.lane_encoder_.transform(self._lane_codes(df)))

    # ---- prediction ---------------------------------------------------------------
    def _raw_log_rpm(self, X: pd.DataFrame) -> np.ndarray:
        design = time_design(X, self.mean_log_distance_)
        return self.gbm_.predict(self._static(X)) + design @ self.time_coef_

    def predict(self, df: pd.DataFrame, market_index_by_day: pd.Series) -> np.ndarray:
        X = build_features(df, market_index_by_day)
        return np.exp(self._raw_log_rpm(X) + self._lane_term(df)) * df["distance"].to_numpy()

    def components(self, df: pd.DataFrame, market_index_by_day: pd.Series) -> pd.DataFrame:
        """The prediction split into its additive log-scale parts (for plots and sanity checks)."""
        X = build_features(df, market_index_by_day)
        design = time_design(X, self.mean_log_distance_)
        coef = self.time_coef_
        return pd.DataFrame({
            "static": self.gbm_.predict(self._static(X)),
            "calendar": design[:, :-2] @ coef[:-2],
            "market": design[:, -2:] @ coef[-2:],
            "lane": self._lane_term(df) * np.ones(len(df)),
        }, index=df.index)

    def time_effects(self) -> pd.Series:
        return pd.Series(self.time_coef_, index=TIME_TERMS)
