"""Draw the figures used in the report -> reports/figures/.

    python scripts/make_figures.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight_rate.data import (EQUIPMENT, city_coordinates, daily_market_index, load_december_inputs,
                               load_development, load_validation, with_coordinates)
from freight_rate.baselines import KitchenSinkGBM
from freight_rate.diagnostics import quote_signal_correlations
from freight_rate.model import FreightRateModel
from freight_rate.validation import corruption_flags

FIG = ROOT / "reports" / "figures"
TEAL, ORANGE, GREY, RED = "#064A56", "#D9822B", "#8A9BA0", "#B23A48"
EQUIP_COLOR = dict(zip(EQUIPMENT, [TEAL, ORANGE, "#6C8EAD"]))
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "figure.dpi": 150})


def style(ax):
    ax.grid(axis="y", color="#E3EAEC", linewidth=0.8)
    ax.set_axisbelow(True)


def level_figure(dev, val, model, market, keep):
    """Daily level of the rate (static part and market effect removed) with the fitted calendar term."""
    parts = model.components(dev, market)
    y = np.log(dev["posted_rate"] / dev["distance"])
    level = (y - parts["static"] - parts["market"] - parts["lane"])[keep]
    daily = level.groupby([dev["date"][keep], dev["equipment"][keep]]).mean().unstack()

    days = pd.date_range("2025-01-01", "2025-12-31")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for eq in EQUIPMENT:
        ax.plot(daily.index, 100 * daily[eq], ".", ms=2.6, color=EQUIP_COLOR[eq], alpha=0.55)
        grid = pd.DataFrame({"date": days, "equipment": eq, "distance": 1000.0, "weight": 30000.0,
                             "pickup": "Lexington", "delivery": "Fort Wayne"})
        grid = with_coordinates(grid, city_coordinates(dev, val))
        cal = model.components(grid, market)["calendar"]
        ax.plot(days[days <= "2025-10-31"], 100 * cal[days <= "2025-10-31"], color=EQUIP_COLOR[eq], lw=1.8, label=eq)
        ax.plot(days[days > "2025-10-31"], 100 * cal[days > "2025-10-31"], color=EQUIP_COLOR[eq], lw=1.8, ls=(0, (3, 2)))
    ax.axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), color="#EEF3F4", zorder=0)
    ax.text(pd.Timestamp("2025-11-03"), ax.get_ylim()[1] * 0.92, "validation\nmonths", fontsize=8, color="#455A60", va="top")
    ax.set_title("Rate level after removing lane, load and market effects")
    ax.set_ylabel("% above/below average rate per mile")
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=8.5)
    ax.set_xticks(pd.date_range("2025-01-01", "2025-12-01", freq="2MS"))
    ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    style(ax)

    ax = axes[1]
    for eq in EQUIPMENT:
        ramp = model.time_effects()[f"quarter_end_ramp[{eq}]"]
        ax.plot([0, 1], [0, 100 * ramp], color=EQUIP_COLOR[eq], lw=2, label=f"{eq}: +{100 * ramp:.1f}%")
    ax.set_xticks([0, 1], ["1st of month", "last day"])
    ax.set_title("Quarter-end ramp (Mar, Jun, Sep, Dec)")
    ax.set_ylabel("% added to the rate")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_calendar_level.png", bbox_inches="tight")
    plt.close(fig)


def quote_signal_figure(dev, val, model, market, keep):
    corr = quote_signal_correlations(dev, market, model)
    corr_simple, corr_full = corr["simple"].to_numpy(), corr["full"].to_numpy()

    allrows = pd.concat([dev[["date", "quote_signal"]], val[["date", "quote_signal"]]])
    skew = allrows.groupby(allrows["date"].dt.month)["quote_signal"].skew()
    std = allrows.groupby(allrows["date"].dt.month)["quote_signal"].std()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    ax = axes[0]
    months = np.arange(1, 11)
    ax.bar(months - 0.2, corr_simple, width=0.4, color="#9DB7BE", label="vs. distance + equipment + market")
    ax.bar(months + 0.2, corr_full, width=0.4, color=TEAL, label="vs. full model's error")
    ax.text(11.5, 0.03, "labels\nhidden", ha="center", fontsize=8, color="#455A60")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(range(1, 13), MONTHS, fontsize=8)
    ax.set_title("Correlation of quote_signal with the rate")
    ax.set_ylabel("within-month correlation")
    ax.set_ylim(-0.4, 0.62)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center")
    style(ax)
    ax = axes[1]
    ax.bar(skew.index, skew.values, color=[GREY if m in (8, 11, 12) else ORANGE for m in skew.index])
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(range(1, 13), MONTHS, fontsize=8)
    ax.set_title("Skewness of quote_signal")
    style(ax)
    ax = axes[2]
    ax.bar(std.index, std.values, color=[GREY if m in (8, 11, 12) else ORANGE for m in std.index])
    ax.set_xticks(range(1, 13), MONTHS, fontsize=8)
    ax.set_title("Spread of quote_signal")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_quote_signal_regimes.png", bbox_inches="tight")
    plt.close(fig)


def corruption_figure(dev, model, market):
    parts = model.components(dev, market)
    err = np.log(dev["posted_rate"] / dev["distance"]) - parts.sum(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.hist(err, bins=160, color=TEAL)
    ax.set_yscale("log")
    ax.set_xlabel("log(actual rate / model rate)")
    ax.set_ylabel("loads (log scale)")
    ax.set_title("Most labels agree with the model; 1.4% are off by 2x-5x")
    style(ax)
    ax = axes[1]
    ax.hist(err[err.abs() < 0.25], bins=70, color=TEAL)
    ax.set_xlabel("log error on the other 98.6%")
    ax.set_title(f"Core noise: sd = {err[err.abs() < 0.25].std():.3f}")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_label_corruption.png", bbox_inches="tight")
    plt.close(fig)


def forecast_diagnostics(dev, market, corrupt):
    folds = [("Apr-May", "2025-04-01", "2025-06-01"), ("May-Jun", "2025-05-01", "2025-07-01"),
             ("Jun-Jul", "2025-06-01", "2025-08-01"), ("Jul-Aug", "2025-07-01", "2025-09-01"),
             ("Aug-Sep", "2025-08-01", "2025-10-01"), ("Sep-Oct", "2025-09-01", "2025-11-01")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw={"width_ratios": [1, 1.5]})
    bias_rms = {}
    palette = ["#9DB7BE", "#6FA0AC", "#3F8394", "#1F6475", TEAL, ORANGE]
    for (name, a, b), color in zip(folds, palette):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        test_mask = ((dev["date"] >= a) & (dev["date"] < b)).to_numpy() & ~corrupt
        model = FreightRateModel().fit(dev[dev["date"] < a], market)
        test = dev[test_mask]
        pred = model.predict(test, market)
        err = np.log(test["posted_rate"].to_numpy() / pred)
        daily = pd.Series(err, index=test["date"]).groupby(level=0).mean()
        bias_rms[name] = float(np.sqrt((daily ** 2).mean()))
        axes[1].plot(daily.index, 100 * daily.values, color=color, lw=1.4, label=name)
        if name == "Sep-Oct":
            sink = KitchenSinkGBM().fit(dev[dev["date"] < a], market)
            sink_err = np.log(test["posted_rate"].to_numpy() / sink.predict(test, market))
            ax = axes[0]
            bins = np.linspace(-0.12, 0.12, 97)
            ax.hist(100 * (np.exp(sink_err) - 1), bins=100 * (np.exp(bins) - 1), color=GREY, alpha=0.85,
                    label="boosted trees on raw columns")
            ax.hist(100 * (np.exp(err) - 1), bins=100 * (np.exp(bins) - 1), color=TEAL, alpha=0.9, label="final model")
            ax.set_xlabel("error: actual vs predicted rate (%), beyond +/-12% not shown")
            ax.set_ylabel("loads")
            ax.set_title("Sep-Oct loads, trained on Jan-Aug")
            ax.legend(frameon=False, fontsize=8)
            style(ax)
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_ylabel("daily mean error (%)")
    axes[1].set_title("Out-of-time bias by day (every fold)")
    axes[1].legend(frameon=False, fontsize=8, ncol=3)
    axes[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    style(axes[1])
    fig.tight_layout()
    fig.savefig(FIG / "fig4_out_of_time_diagnostics.png", bbox_inches="tight")
    plt.close(fig)
    out = ROOT / "outputs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    pd.Series(bias_rms, name="daily bias rms").rename_axis("fold").to_csv(out / "daily_bias_by_fold.csv")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    dev, val = load_development(), load_validation()
    market = daily_market_index(dev, val)
    model = FreightRateModel().fit(dev, market)
    keep = model.train_keep_
    level_figure(dev, val, model, market, keep)
    quote_signal_figure(dev, val, model, market, keep)
    corruption_figure(dev, model, market)
    forecast_diagnostics(dev, market, ~keep)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
