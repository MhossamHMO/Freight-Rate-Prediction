"""Small guards around the places where this problem is easy to get wrong.

Run with:  python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from freight_rate.data import (build_features, city_coordinates, daily_market_index,
                               load_december_inputs, load_development, load_validation,
                               quarter_end_progress, with_coordinates)
from freight_rate.model import STATIC_FEATURES, FreightRateModel


@pytest.fixture(scope="module")
def frames():
    dev, val = load_development(), load_validation()
    return dev, val, daily_market_index(dev, val)


def test_validation_is_strictly_after_development(frames):
    dev, val, _ = frames
    assert dev["date"].max() < val["date"].min()


def test_features_clean_weight_and_fill_market_index(frames):
    dev, _, market = frames
    X = build_features(dev, market)
    assert (X["weight"].dropna() > 0).all()                 # sign flips removed
    assert np.isfinite(X["log_market"]).all()               # missing market_index filled by day
    assert "quote_signal" not in STATIC_FEATURES


def test_quarter_end_progress():
    dates = pd.Series(pd.to_datetime(["2025-02-28", "2025-03-01", "2025-03-31", "2025-12-16"]))
    np.testing.assert_allclose(quarter_end_progress(dates), [0.0, 0.0, 1.0, 15 / 30])


def test_december_rows_get_coordinates_from_loads_data(frames):
    dev, val, _ = frames
    rows = with_coordinates(load_december_inputs(), city_coordinates(dev, val))
    assert rows[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].notna().all().all()


def test_model_recovers_structure_and_ignores_corrupted_labels(frames):
    dev, val, market = frames
    sample = dev[dev["date"] < "2025-06-01"]
    model = FreightRateModel().fit(sample, market)
    assert 0.010 < model.n_flagged_ / len(sample) < 0.020          # ~1.4% corrupted labels
    assert model.residual_sd_ < 0.025
    coef = model.time_effects()
    assert coef["quarter_end_ramp[Flatbed]"] > coef["quarter_end_ramp[Reefer]"] > coef["quarter_end_ramp[Dry Van]"] > 0
    pred = model.predict(val.head(500), market)
    assert np.isfinite(pred).all() and (pred > 0).all()
