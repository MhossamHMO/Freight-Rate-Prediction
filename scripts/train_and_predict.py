"""Train on all development data and write the two prediction files.

    python scripts/train_and_predict.py

Writes validation_predictions.csv (repo root) and fills predicted_rate in
data/december_chart_inputs.csv, ready for score.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from freight_rate.data import (DATA_DIR, city_coordinates, daily_market_index, load_december_inputs,
                               load_development, load_validation, with_coordinates)
from freight_rate.model import FreightRateModel


def main() -> None:
    dev, val, dec = load_development(), load_validation(), load_december_inputs()
    market = daily_market_index(dev, val)       # features only; covers every December day

    model = FreightRateModel().fit(dev, market)
    print(f"Trained on {len(dev):,} loads; {model.n_flagged_} corrupted labels left out; "
          f"residual sd on clean rows = {model.residual_sd_:.4f} (log scale)")
    print(model.time_effects().round(4).to_string())

    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    predicted = pd.Series(model.predict(val, market).round(2), index=val["load_id"])
    submission = pd.DataFrame({"load_id": template["load_id"],
                               "predicted_rate": template["load_id"].map(predicted)})
    assert submission["predicted_rate"].notna().all() and (submission["predicted_rate"] > 0).all()
    submission.to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"Wrote validation_predictions.csv ({len(submission):,} rows)")

    december = pd.read_csv(DATA_DIR / "december_chart_inputs.csv")
    december_rows = with_coordinates(dec, city_coordinates(dev, val))   # inputs carry names only
    december["predicted_rate"] = model.predict(december_rows, market).round(2)
    december.to_csv(DATA_DIR / "december_chart_inputs.csv", index=False)
    print("Filled predicted_rate in data/december_chart_inputs.csv")


if __name__ == "__main__":
    main()
