# Freight Rate Prediction Challenge

Predicts the posted rate of a truckload from its lane, equipment, weight, date and market index.
Train/validate on `data/train_test.csv` (Jan-Oct 2025), predict the 12,000 loads in
`data/validation.csv` (Nov-Dec 2025), and fill the fixed December chart rows.
The write-up (approach, validation, December chart) is `reports/report.pdf`.

## Run it

```bash
python -m pip install -r requirements.txt

python scripts/train_and_predict.py          # -> validation_predictions.csv, fills data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
                                             # -> scorer_results/candidate_december.png

python scripts/run_validation.py             # validation tables  -> outputs/validation/   (~10 min)
python scripts/make_figures.py               # report figures     -> reports/figures/
python scripts/build_report.py               # PDF report         -> reports/report.pdf
python -m pytest -q                          # sanity tests
```

Python 3.10+ (developed on 3.14). Everything is deterministic; no GPU or network needed.

## What the model does

```
log(rate / distance) = G(distance, equipment, weight, geography)        gradient-boosted trees
                     + T(day, quarter-end ramp by equipment, log market) linear, extrapolates in time
                     + shrunk pickup / delivery / lane offsets            ridge, for cities seen in training
```

* **Out-of-time by construction.** The validation months come after all training data, so every
  decision was judged on expanding-window folds (train on everything before, test on the next two
  months), not on a random split.
* **Corrupted labels are removed, not fitted.** About 1.4% of training rates are off by a factor of
  2-5x in either direction. They are unrelated to any feature, so they are detected from the fit and
  excluded from training (677 rows).
* **`quote_signal` is not used.** Its relationship with the rate flips sign from month to month and
  disappears in the months it resembles the validation period; see the report.
* **Unseen cities** (8 of the 72 cities in the validation set never appear in training) are handled
  through their coordinates; the lane offsets simply do not apply to them.

## Layout

| Path | What it is |
|---|---|
| `src/freight_rate/data.py` | loading, cleaning decisions, feature construction, city coordinate lookup |
| `src/freight_rate/model.py` | the two-part model (back-fitting, corrupted-label filter, lane correction) |
| `src/freight_rate/validation.py` | out-of-time folds, unseen-city folds, metrics |
| `src/freight_rate/baselines.py` | reference models used for comparison |
| `scripts/` | train + predict, run validation, make figures, build the report |
| `data/` | the provided inputs (`december_chart_inputs.csv` has `predicted_rate` filled in) |
| `validation_predictions.csv` | the submission file (`load_id,predicted_rate`) |
| `outputs/validation/` | validation tables referenced in the report |
| `reports/` | `report.pdf` and the `figures/` used in it |
