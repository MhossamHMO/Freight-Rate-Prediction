"""Assemble reports/report.pdf from the validation tables, figures and scorer chart.

    python scripts/build_report.py

Run after train_and_predict.py, score.py, run_validation.py and make_figures.py.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from freight_rate.data import (city_coordinates, daily_market_index, load_december_inputs, load_development,
                               load_validation, with_coordinates)
from freight_rate.diagnostics import quote_signal_correlations
from freight_rate.model import FreightRateModel

VAL = ROOT / "outputs" / "validation"
FIG = ROOT / "reports" / "figures"
TEAL = colors.HexColor("#064A56")
PALE = colors.HexColor("#EEF3F4")

base = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=base["Normal"], fontName="Helvetica", fontSize=9.6, leading=13.2,
                      spaceAfter=5, alignment=TA_LEFT)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.4, leading=11, textColor=colors.HexColor("#455A60"))
H1 = ParagraphStyle("h1", parent=BODY, fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=TEAL,
                    spaceBefore=11, spaceAfter=5)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Helvetica-Bold", fontSize=10.5, leading=14, spaceBefore=6,
                    spaceAfter=3)
TITLE = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=TEAL,
                       spaceAfter=3)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=13, bulletIndent=2, spaceAfter=3)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.3, leading=10.4, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, BULLET, bulletText="•") for t in items]


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C5D1D4")),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), PALE)]
    t.setStyle(TableStyle(style))
    return t


def figure(name, width=6.9, caption=None):
    path = name if isinstance(name, Path) else FIG / name
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    items = [Image(str(path), width=width * inch, height=width * inch * h / w)]
    if caption:
        items.append(p(caption, SMALL))
    return KeepTogether(items)


def pct(x, digits=1):
    return f"{x:.{digits}f}%"


def pooled_ramps(dev, market, model) -> dict:
    """Size of the observed ramp in each quarter-end month, all equipment pooled (percent)."""
    parts = model.components(dev, market)
    y = np.log(dev["posted_rate"] / dev["distance"])
    drift = model.time_effects()["drift_per_100d"] * (dev["date"] - pd.Timestamp("2025-01-01")).dt.days / 100
    level = (y - parts["static"] - parts["market"] - parts["lane"] - drift)[model.train_keep_]
    out = {}
    for month, name in ((3, "March"), (6, "June"), (9, "September")):
        sel = (dev["date"].dt.month == month)[model.train_keep_]
        progress = ((dev["date"].dt.day - 1) / (dev["date"].dt.days_in_month - 1))[model.train_keep_][sel]
        out[name] = 100 * np.polyfit(progress, level[sel], 1)[0]
    return out


def main() -> None:
    dev, val = load_development(), load_validation()
    market = daily_market_index(dev, val)
    model = FreightRateModel().fit(dev, market)
    coef = model.time_effects()

    summary = pd.read_csv(VAL / "temporal_summary.csv", index_col=0)
    by_fold = pd.read_csv(VAL / "temporal_log_rmse_by_fold.csv", index_col=0)
    final_folds = pd.read_csv(VAL / "temporal_folds_final.csv", index_col=0)
    rand = pd.read_csv(VAL / "random_vs_temporal.csv", index_col=0)
    unseen = pd.read_csv(VAL / "unseen_cities.csv", index_col=[0, 1])
    s_final = summary.loc["Final model"]
    s_sink = summary.loc["Boosted trees on all raw columns"]
    s_med = summary.loc["Median rate-per-mile by equipment"]

    seen_cities = set(dev["pickup"]) | set(dev["delivery"])
    new_cities = sorted((set(val["pickup"]) | set(val["delivery"])) - seen_cities)
    new_rows = int((~val["pickup"].isin(seen_cities) | ~val["delivery"].isin(seen_cities)).sum())
    keep = model.train_keep_
    mult = np.exp(np.log(dev["posted_rate"] / dev["distance"]) - model.components(dev, market).sum(axis=1))[~keep]
    err_all = np.log(dev["posted_rate"] / dev["distance"]) - model.components(dev, market).sum(axis=1)
    max_clean_err, min_bad_err = err_all[keep].abs().max(), err_all[~keep].abs().min()
    bias = pd.read_csv(VAL / "daily_bias_by_fold.csv", index_col=0)["daily bias rms"]
    ramps = pooled_ramps(dev, market, model)
    qs_corr = quote_signal_correlations(dev, market, model)
    dec = pd.read_csv(ROOT / "data" / "december_chart_inputs.csv")
    dec_pred = dec["predicted_rate"]

    story = []
    story += [p("Freight Rate Prediction", TITLE),
              p("Approach, validation and the fixed December chart &nbsp;|&nbsp; Machine Learning Engineer assessment", SMALL),
              Spacer(1, 6)]

    # ------------------------------------------------------------------ 1 summary
    story += [p("1. Summary", H1)]
    story += [p(
        f"The task is to price {len(val):,} loads for November and December 2025 from {len(dev):,} labelled loads "
        f"from January to October. The validation loads therefore lie entirely in the future of the training data, "
        f"and {len(new_cities)} of the {len(set(val['pickup']) | set(val['delivery']))} cities in them never appear in training. "
        f"I built the validation around those two facts instead of using a random split.")]
    story += bullets([
        f"<b>Model.</b> log(rate per mile) = a gradient-boosted tree on distance, equipment, weight and geography "
        f"+ a small linear part for the calendar and market index + shrunk offsets for known cities and lanes. "
        f"The calendar part is linear so it extrapolates into months the model has not seen.",
        f"<b>Accuracy, out of time.</b> Over six expanding-window folds (train on everything earlier, predict the next "
        f"two months) the typical error is {pct(s_final['clean MAPE (%)'])} MAPE / ${s_final['clean MAE ($)']:.0f} MAE on "
        f"correctly labelled loads, against {pct(s_sink['clean MAPE (%)'])} / ${s_sink['clean MAE ($)']:.0f} for boosted trees "
        f"on the raw columns and {pct(s_med['clean MAPE (%)'])} / ${s_med['clean MAE ($)']:.0f} for a median rate-per-mile rule.",
        f"<b>Label noise.</b> {int((~keep).sum())} training rates ({pct(100 * (~keep).mean(), 2)}) are corrupted: off by a "
        f"factor of {mult[mult > 1].min():.1f}-{mult[mult > 1].max():.1f} or {mult[mult < 1].min():.2f}-{mult[mult < 1].max():.2f}, with no link to any feature. "
        f"They are removed from training. Any metric computed on all rows, including corrupted ones, is "
        f"limited to roughly {pct(s_final['all-rows MAPE (%)'])} MAPE for this reason, whatever the model.",
        "<b>Two columns could not be taken at face value.</b> quote_signal relates to the rate with a sign that flips "
        "from month to month (mostly because it tracks the time level) and looks inactive in the validation months, so it is excluded; "
        "market_index is replaced by its daily average because its per-load noise carries no information.",
        "<b>December.</b> The fixed Lexington to Fort Wayne chart rises from "
        f"${dec_pred.iloc[0]:.0f} to ${dec_pred.iloc[-1]:.0f} through the month: a quarter-end ramp, a slow drift and a weekly "
        "cycle from the market index (section 6)."])

    # ------------------------------------------------------------------ 2 findings
    story += [p("2. What the data showed", H1)]
    story += bullets([
        f"<b>The rate is almost a clean product of a few effects.</b> After the model's terms, the remaining noise "
        f"on correctly labelled loads is about {100 * model.residual_sd_:.1f}% (standard deviation of the log error), "
        "and its tails are Gaussian. Rate per mile falls with distance, Reefer and Flatbed carry premiums of about 12% and 8% over Dry Van, "
        "the rate per mile rises with weight (about 9% from the lightest to the heaviest loads, flattening above 40,000 lb), and the market index "
        f"has an elasticity of about {coef['market_elasticity']:.2f}.",
        "<b>Geography is a smooth, mostly additive effect.</b> A city adds roughly the same premium as a pickup "
        "and as a delivery (correlation 0.99 between the two), running from about -3% in the Great Lakes and Northeast to "
        "+3% in Texas and Oklahoma. Predicting a held-out city's premium from its three nearest cities' coordinates "
        "recovers about 88% of its variance, which is what lets the model price unseen cities.",
        f"<b>The calendar matters, in two ways.</b> There is a slow upward drift (about {100 * coef['drift_per_100d']:.1f}% per "
        f"100 days) and a ramp in the last month of each quarter. During March, June and September the rate climbs "
        f"by {100 * coef['quarter_end_ramp[Dry Van]']:.1f}% (Dry Van), {100 * coef['quarter_end_ramp[Reefer]']:.1f}% (Reefer) and "
        f"{100 * coef['quarter_end_ramp[Flatbed]']:.1f}% (Flatbed) between the 1st and the last day, "
        "then resets at the start of the next quarter (roughly 1 : 2 : 3 across the three equipment types). December is the next "
        "quarter-end month. I found no holiday effect: the days around Memorial Day, July 4 and Labor Day "
        "move the daily level by less than 1%, in line with ordinary day-to-day noise, and load volumes do not dip either."])
    story += [figure("fig1_calendar_level.png", 6.9,
                     "Figure 1. Left: daily average of the rate after removing the model's lane, load and market effects, "
                     "with the fitted calendar term (dashed after October). Right: size of the quarter-end ramp.")]

    # ------------------------------------------------------------------ 3 data quality
    story += [p("3. Data-quality issues and how I handled them", H1)]
    wneg = (dev["weight"] < 0).sum(), (val["weight"] < 0).sum()
    wna = dev["weight"].isna().sum(), val["weight"].isna().sum()
    mna = dev["market_index"].isna().sum(), val["market_index"].isna().sum()
    capped = (dev["weight"].abs() == 47500).sum()
    rows = [["Issue", "Evidence (development / validation)", "Handling"],
            ["Corrupted rates",
             f"{int((~keep).sum())} rows ({pct(100 * (~keep).mean(), 2)}); {int((mult > 1).sum())} are x{mult[mult > 1].min():.1f}-{mult[mult > 1].max():.1f} too high and "
             f"{int((mult < 1).sum())} are x{mult[mult < 1].min():.2f}-{mult[mult < 1].max():.2f} too low. Spread evenly over months, equipment and distance (Fig. 2).",
             f"Detected from the fit (error beyond 0.25 in log terms; the largest error on any other row is {max_clean_err:.2f}, "
             f"the smallest on a corrupted row {min_bad_err:.2f}) and left out of training. "
             "Reported separately: metrics on clean rows and on all rows."],
            ["quote_signal is unstable",
             f"Within each month it correlates with the rate error, but the sign changes: + in Jan, Feb, Mar, Jun, Sep; - in Apr, May, Jul, Oct; "
             f"about 0 in Aug. Against a simple baseline (distance, equipment, market index) |r| is {qs_corr['simple'].abs()[[1,2,3,4,5,6,7,9,10]].min():.2f}-{qs_corr['simple'].abs().max():.2f}; "
             f"once the calendar level is modelled it falls to {qs_corr['full'].abs()[[1,2,3,4,5,6,7,9,10]].min():.2f}-{qs_corr['full'].abs().max():.2f} with the same signs, because it "
             "mostly tracks the time level. Its shape follows the sign (right-skewed, left-skewed, symmetric), and Nov and Dec are symmetric like Aug (Fig. 3).",
             "Not used. Adding it to the final model raises out-of-time error from "
             f"{summary.loc['Final model', 'clean log RMSE']:.3f} to {summary.loc['Final model + quote_signal', 'clean log RMSE']:.3f} (worst fold: {by_fold['Final model + quote_signal'].max():.2f})."],
            ["market_index noise and gaps",
             f"Per-load value = daily level + noise (sd 0.025), and that noise has no relation to the rate (correlation 0.003). Missing: {mna[0]} / {mna[1]}.",
             "Replaced by the mean of that day's loads. Also provides the market level for the December chart dates."],
            ["Negative weights", f"{wneg[0]} / {wneg[1]} loads. Their magnitudes match the positive ones.",
             "Absolute value. The model's error on these rows equals its error elsewhere."],
            ["Missing weight", f"{wna[0]} / {wna[1]} loads; no bias in the model's error on them.", "Filled with the training median."],
            ["Clipped values", f"Weight capped at 47,500 ({capped} loads, 2.5%); distance floored at 70 miles (48 / 21 loads).",
             "Kept as given; the validation set is clipped the same way."],
            ["Cities missing from training", f"{len(new_cities)} cities ({', '.join(new_cities)}), {new_rows:,} validation loads ({100 * new_rows / len(val):.0f}%).",
             "Priced from coordinates; lane offsets give them no correction. Tested explicitly (section 4)."],
            ["Synthetic coordinates", "Positions are internally consistent per city but not real (e.g. Los Angeles at 28.6 N), and distance cannot be recomputed from them.",
             "Used only as relative position; distance taken as given."],
            ["December chart rows lack inputs", "The chart file has city names, no coordinates and no market index.",
             "Coordinates looked up from the loads data; market index is each day's average from validation.csv (features only)."]]
    story += [table(rows, [1.25, 3.2, 2.45])]
    story += [Spacer(1, 4), figure("fig3_label_corruption.png", 6.7,
              "Figure 2. Error of the final model on every training load. The two small humps at about +1.2 and -1.3 are the corrupted labels.")]
    story += [figure("fig2_quote_signal_regimes.png", 6.7,
              "Figure 3. Left: within-month correlation of quote_signal with the rate error (labels are not available for Nov and Dec). "
              "Middle and right: the shape of quote_signal itself, which in Nov and Dec matches August, the month with no effect.")]

    # ------------------------------------------------------------------ 4 validation
    story += [p("4. Train / validation split and how I validated", H1)]
    story += [p("<b>Why not a random split.</b> The final test is the two months after the last training day and includes new "
                "cities. A shuffled split lets the model see the same days and the same lanes on both sides, so it rewards memorisation "
                "of month-specific behaviour, which is exactly what fails on the validation months:")]
    r = [["Model", "Shuffled 5-fold (log RMSE)", "Out-of-time (log RMSE)", "Shuffled MAPE", "Out-of-time MAPE"]]
    for name, row in rand.iterrows():
        r.append([name, f"{row['random 5-fold log RMSE']:.4f}", f"{row['out-of-time log RMSE']:.4f}",
                  pct(row["random 5-fold MAPE (%)"], 2), pct(row["out-of-time MAPE (%)"], 2)])
    story += [table(r, [2.3, 1.35, 1.35, 0.95, 0.95])]
    story += [Spacer(1, 4), p(
        "<b>What I used instead.</b>")]
    story += bullets([
        "<b>Expanding-window out-of-time folds.</b> Six folds. Each trains on every development row before a cut-off and tests "
        "on the following two months (cut-offs 1 Apr, 1 May, 1 Jun, 1 Jul, 1 Aug, 1 Sep), the same shape as the real task. "
        f"Training sizes run from {int(final_folds.loc['Apr-May', 'n_train']):,} to {int(final_folds.loc['Sep-Oct', 'n_train']):,} loads. "
        "Four of the six test windows contain a quarter-end ramp, which checks that the ramp learned from earlier quarters carries forward.",
        "<b>Unseen-city test.</b> Eight random cities (the same count as in the real validation set) are removed from training "
        "entirely, and the model is scored on September-October loads that touch them. The same rows are scored by a model that did "
        "see those cities as a control. Five random draws.",
        "<b>Clean and all-row metrics.</b> The corrupted rows are flagged once from a reference fit and held fixed, so every "
        "candidate is judged on the same rows. I report both: 'clean' shows what the model can control, 'all rows' is what a scorer using raw labels will see.",
        "<b>No tuning on the final set.</b> It has no labels. Design choices were made on the fold average, and the boosting "
        "settings were barely sensitive (during development, log RMSE stayed between 0.0196 and 0.0201 across the settings I tried), so there is little to over-fit.",
        "The submitted model is the same pipeline refit on all 48,000 development loads."])
    t = [["Out-of-time results, mean of 6 folds", "Log RMSE", "MAPE (clean)", "MAE (clean)", "MAPE (all rows)"]]
    for name, row in summary.iterrows():
        label = f"<b>{name}</b>" if name == "Final model" else name
        t.append([label, f"{row['clean log RMSE']:.4f}", pct(row["clean MAPE (%)"], 2), f"${row['clean MAE ($)']:.1f}",
                  pct(row["all-rows MAPE (%)"], 2)])
    story += [Spacer(1, 3), table(t, [2.9, 0.9, 1.0, 1.0, 1.1])]
    story += [p("Log RMSE is the typical error as a fraction of the rate (0.019 is about 1.9%). The no-geography row is close to the "
                "final model here because the city and lane offsets already cover every city seen in training; coordinates matter for cities "
                "that were not seen (below).", SMALL)]

    fold_tbl = [["Final model by fold", *final_folds.index]]
    fold_tbl.append(["Log RMSE"] + [f"{v:.4f}" for v in final_folds["clean log RMSE"]])
    fold_tbl.append(["MAPE (clean)"] + [pct(v, 2) for v in final_folds["clean MAPE (%)"]])
    fold_tbl.append(["MAE (clean)"] + [f"${v:.1f}" for v in final_folds["clean MAE ($)"]])
    fold_tbl.append(["MAPE (all rows)"] + [pct(v, 2) for v in final_folds["all-rows MAPE (%)"]])
    story += [Spacer(1, 4), table(fold_tbl, [1.5, *[0.9] * len(final_folds)])]

    u = unseen["Final model"].unstack()
    ug = unseen["Final model without geography"].unstack()
    story += [Spacer(1, 4), p(
        f"<b>Unseen cities.</b> With the cities removed from training, the final model's log RMSE is {u.loc['unseen', 'clean log RMSE']:.4f} "
        f"(MAPE {u.loc['unseen', 'clean MAPE (%)']:.2f}%), against {u.loc['seen', 'clean log RMSE']:.4f} for the same rows when the cities were "
        f"known. Without coordinates it would be {ug.loc['unseen', 'clean log RMSE']:.4f}: the geography features are what make new cities "
        f"priceable. Only {100 * new_rows / len(val):.0f}% of the validation loads involve a new city.")]
    story += [figure("fig4_out_of_time_diagnostics.png", 6.9,
              "Figure 4. Left: error on September-October loads for models trained on January-August. "
              "Right: the daily mean error in every out-of-time fold (the level drifts, but stays within about 2%).")]

    # ------------------------------------------------------------------ 5 model
    story += [p("5. Model and the reasoning behind it", H1)]
    story += [p(
        "<b>Target.</b> log(rate / distance). Rate scales with distance almost proportionally, so the rate per mile is the "
        "natural quantity, and in logs the effects are additive, which is what the data shows.")]
    story += bullets([
        "<b>G: gradient-boosted trees</b> (scikit-learn <i>HistGradientBoostingRegressor</i>) on distance, equipment, weight and geography "
        "(pickup and delivery coordinates, their midpoint and difference). Nothing in G depends on the date, so it cannot memorise a "
        "month. This is where the non-linear distance, weight and regional structure is learned.",
        "<b>T: a linear calendar-and-market part</b> with an intercept, a daily drift, a quarter-end ramp per equipment type, "
        "log market index and market index x log distance. Trees cannot extrapolate: a tree given the date would predict "
        "the last October level for every later day. A linear term continues the drift and applies the Mar/Jun/Sep ramp to December.",
        "<b>Back-fitting.</b> G and T are fitted alternately (four rounds) so that neither absorbs the other's signal; corrupted labels are "
        "detected after the first pass and the fit is repeated without them.",
        "<b>Shrunk city and lane offsets.</b> A ridge regression on pickup city, delivery city and directed lane, fitted on what G and T leave "
        "behind. It improves every fold, and cities or lanes that were never seen simply receive no offset.",
        "<b>Why not a bigger model.</b> The table above shows the structure, not model capacity, is what matters: the raw-column boosted-tree "
        f"model has more freedom than mine and is {s_sink['clean log RMSE'] / s_final['clean log RMSE']:.0f} times less accurate out of time "
        f"({s_sink['clean log RMSE']:.3f} against {s_final['clean log RMSE']:.3f}). "
        "Tuning the boosting settings moved the result by about 0.0005."])
    te = [["Fitted calendar and market terms (log scale)", "Value"]]
    labels = {"intercept": "Intercept", "drift_per_100d": "Drift per 100 days",
              "quarter_end_ramp[Dry Van]": "Quarter-end ramp, Dry Van", "quarter_end_ramp[Reefer]": "Quarter-end ramp, Reefer",
              "quarter_end_ramp[Flatbed]": "Quarter-end ramp, Flatbed", "market_elasticity": "Market index elasticity",
              "market_elasticity_x_log_distance": "Elasticity change per unit of log distance"}
    for k, v in coef.items():
        te.append([labels[k], f"{v:+.4f}"])
    story += [Spacer(1, 3), table(te, [3.6, 1.0])]

    # ------------------------------------------------------------------ 6 december
    story += [p("6. The fixed December chart", H1)]
    chart = ROOT / "scorer_results" / "candidate_december.png"
    story += [figure(chart, 6.6, "Figure 5. candidate_december.png, produced by the provided score.py from the model's predictions.")]
    lane = dev[(dev["pickup"] == "Lexington") & (dev["delivery"] == "Fort Wayne") & (dev["equipment"] == "Dry Van") & keep]
    dec_rows = with_coordinates(load_december_inputs(), city_coordinates(dev, val))
    market_part = model.components(dec_rows, market)["market"]
    weekly = 100 * (np.exp(market_part.max() - market_part.min()) - 1) / 2
    story += [p(
        f"Only the date varies (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb). The predicted rate goes from ${dec_pred.iloc[0]:.0f} "
        f"on 1 December to ${dec_pred.iloc[-1]:.0f} on 31 December, a rise of {100 * (dec_pred.iloc[-1] / dec_pred.iloc[0] - 1):.1f}%. "
        f"Three things shape it: the quarter-end ramp for Dry Van ({100 * coef['quarter_end_ramp[Dry Van]']:.1f}% across the month), the drift, "
        f"and a weekly cycle of about +/-{weekly:.1f}% that comes from the market index (lowest around Sunday and Monday, highest mid-week). "
        f"The range is ${dec_pred.min():.0f} to ${dec_pred.max():.0f}. For context, the {len(lane)} Dry Van loads on this lane in the development "
        f"data (distance {lane['distance'].min():.0f}-{lane['distance'].max():.0f} miles) paid ${lane['posted_rate'].min():.0f} to ${lane['posted_rate'].max():.0f}, "
        f"median ${lane['posted_rate'].median():.0f}.")]
    story += [p(
        "The chart needs two inputs the file does not contain: city coordinates, taken from the loads data, and the market index for each "
        "December day, taken as the average of that day's loads in validation.csv. Both are features, not labels. "
        "I did not add a Christmas dip or any other holiday shape, because no holiday effect is visible in the January-October data.")]

    # ------------------------------------------------------------------ 7 risks
    story += [p("7. Assumptions and risks", H1)]
    story += bullets([
        f"<b>The December ramp rests on three earlier quarter-ends</b> (March, June, September), which agree closely "
        f"(pooled over equipment: {ramps['March']:.1f}%, {ramps['June']:.1f}% and {ramps['September']:.1f}%, the same ramp within about half a point). If December behaves "
        "differently, the end-of-month predictions are the part that moves.",
        f"<b>Level drift is the main uncertainty.</b> The drift is estimated as a straight line. Out of time, the day-by-day error in the "
        f"overall level was {100 * bias.min():.1f}% to {100 * bias.max():.1f}% (rms) by fold, worst in {bias.idxmax()}, when the drift paused. "
        "During development, anchoring the level to the latest weeks or freezing the drift did not do better, so I kept the simple version.",
        "<b>quote_signal in Nov-Dec is an inference, not a measurement.</b> I cannot see the validation labels. The evidence is that its "
        "distribution matches August, the one month with no effect, and that using it is costly when the month's regime changes. If labels "
        "were available I would test it on the first weeks of November before trusting it either way.",
        f"<b>The validation labels probably contain the same kind of corrupted rates.</b> At the training rate of 1.4%, even a near-perfect "
        f"model scores about {s_final['all-rows MAPE (%)'] - s_final['clean MAPE (%)']:.1f} percentage points worse on raw labels "
        f"({pct(s_final['all-rows MAPE (%)'])} against {pct(s_final['clean MAPE (%)'])} here). Squared-error metrics will be hit harder than absolute ones.",
        "<b>Not modelled:</b> any effect of holidays, weather or events outside the January-October window; changes in the lane mix; interactions of the "
        "ramp with geography (I checked distance and pickup latitude and found none)."])

    # ------------------------------------------------------------------ 8 reproduce
    story += [p("8. Reproducing this work", H1)]
    story += [p("Code, dependencies and instructions are in the repository README. In short: <font face='Courier'>pip install -r requirements.txt</font>, "
                "<font face='Courier'>python scripts/train_and_predict.py</font>, then "
                "<font face='Courier'>python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv</font>. "
                "The validation tables, figures and this report are regenerated by <font face='Courier'>scripts/run_validation.py</font>, "
                "<font face='Courier'>scripts/make_figures.py</font> and <font face='Courier'>scripts/build_report.py</font>.")]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6B7F84"))
        canvas.drawString(0.8 * inch, 0.5 * inch, "Freight rate prediction - approach and validation")
        canvas.drawRightString(letter[0] - 0.8 * inch, 0.5 * inch, f"Page {doc.page}")
        canvas.restoreState()

    out = ROOT / "reports" / "report.pdf"
    SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                      topMargin=0.7 * inch, bottomMargin=0.8 * inch, title="Freight Rate Prediction - Approach and Validation",
                      author="Candidate").build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", out)


if __name__ == "__main__":
    main()
