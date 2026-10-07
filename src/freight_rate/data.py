"""Loading the raw files and turning them into model-ready features.

Everything here is label-free except `load_development`, so the same code path
prepares development rows, validation rows and the fixed December chart rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

EQUIPMENT = ["Dry Van", "Reefer", "Flatbed"]
EQUIPMENT_CODE = {name: i for i, name in enumerate(EQUIPMENT)}
QUARTER_END_MONTHS = (3, 6, 9, 12)
ORIGIN = pd.Timestamp("2025-01-01")

GEO_COLUMNS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


def load_development(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "train_test.csv", parse_dates=["date"])
    return df.sort_values(["date", "load_id"]).reset_index(drop=True)


def load_validation(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "validation.csv", parse_dates=["date"])


def load_december_inputs(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "december_chart_inputs.csv", parse_dates=["date"])


def daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """Mean market_index per calendar day, pooled over whichever frames are given.

    The per-row value is the day's market level plus ~0.025 of independent noise,
    and the noise carries no information about the rate (checked on the clean
    development rows), so the day average is the better signal.  It also fills
    the rows where market_index is missing and supplies a value for the December
    chart rows, which only come with a date.  Uses features only, never labels.
    """
    pooled = pd.concat([f[["date", "market_index"]] for f in frames])
    return pooled.groupby("date")["market_index"].mean().dropna()


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One (lat, lon) row per city, read from the pickup/delivery columns of the given frames.

    The December chart rows only carry city names, so their coordinates come from here.
    Each city has exactly one coordinate pair in the data (checked below).
    """
    parts = []
    for f in frames:
        for role in ("pickup", "delivery"):
            parts.append(f[[role, f"{role}_lat", f"{role}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    table = pd.concat(parts).drop_duplicates()
    if table["city"].duplicated().any():
        raise ValueError("A city appears with more than one coordinate pair")
    return table.set_index("city")


def with_coordinates(df: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Add pickup/delivery coordinates to rows that only have city names."""
    out = df.copy()
    for role in ("pickup", "delivery"):
        lookup = out[role].map
        out[f"{role}_lat"] = lookup(coords["lat"])
        out[f"{role}_lon"] = lookup(coords["lon"])
    if out[GEO_COLUMNS].isna().any().any():
        raise ValueError("Unknown city in input rows")
    return out


def quarter_end_progress(dates: pd.Series) -> np.ndarray:
    """0 outside Mar/Jun/Sep/Dec; inside them, how far through the month we are (0 -> 1)."""
    month = dates.dt.month
    progress = (dates.dt.day - 1) / (dates.dt.days_in_month - 1)
    return np.where(month.isin(QUARTER_END_MONTHS), progress, 0.0)


def build_features(df: pd.DataFrame, market_index_by_day: pd.Series) -> pd.DataFrame:
    """Feature table shared by training, validation and the December chart.

    Cleaning decisions (see the report for the evidence):
      * weight: negative values are sign flips -> absolute value; missing stays NaN
        here and is filled with the training median inside the model.
      * market_index: replaced by the day average (see `daily_market_index`).
      * quote_signal is deliberately not used (its relationship with the rate
        changes sign from month to month and vanishes in the validation months).
    """
    out = pd.DataFrame(index=df.index)
    out["log_distance"] = np.log(df["distance"].astype(float))
    out["equipment"] = df["equipment"].map(EQUIPMENT_CODE).astype(int)
    out["weight"] = df["weight"].abs()
    if "quote_signal" in df:                      # only the validation ablation reads this
        out["quote_signal"] = df["quote_signal"]

    for col in GEO_COLUMNS:
        out[col] = df[col]
    out["lat_mid"] = (df["pickup_lat"] + df["delivery_lat"]) / 2
    out["lon_mid"] = (df["pickup_lon"] + df["delivery_lon"]) / 2
    out["d_lat"] = df["delivery_lat"] - df["pickup_lat"]
    out["d_lon"] = df["delivery_lon"] - df["pickup_lon"]

    mi = df["date"].map(market_index_by_day)
    if mi.isna().any():
        raise ValueError("No market index available for some dates")
    out["log_market"] = np.log(mi.astype(float))

    out["day"] = (df["date"] - ORIGIN).dt.days.astype(float)
    out["quarter_end"] = quarter_end_progress(df["date"])
    return out
