"""Render the one-page case-study figures and HTML from derived Gate 8-10 outputs.

Run from the private etf-monitor checkout root:
    uv run --with matplotlib python build_report.py <out_dir>
Reads only derived outputs under data/ (never re-fetches provider data).
The derived inputs under data/ are produced by the private pipeline and are not distributed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(sys.argv[1])
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
BLUE, RED, GREY = "#1f5fa8", "#c0392b", "#9aa0a6"

# Fig 1: AP response uplift, frozen values from docs/decision-log.md (Gate 10 table).
uplift = [
    ("U.S. high yield (JNK, SJNK)", 18.4, 13.1, 23.4, 16.7, 8.1, 25.3),
    ("EM local bond (EBND)", 25.8, 19.2, 32.9, 24.3, 13.8, 33.3),
    ("Intl gov. bond (BWX)", 20.1, 14.5, 25.6, 13.9, 6.3, 21.9),
    ("U.S. inv. grade (SPAB, SPSB)*", 11.7, 6.1, 17.7, 45.8, 14.0, 58.6),
    ("EM equity (SPEM)", 16.1, 11.3, 20.8, 5.3, 1.7, 8.7),
    ("Dev. intl equity (4 funds)", 7.6, 4.6, 10.8, 8.9, 6.4, 11.4),
    ("U.S. large-cap (SPY, DIA, XL*)", 3.7, 1.5, 6.0, 4.5, 2.3, 6.9),
    ("U.S. small/mid (KRE, XOP, XBI)", 2.2, -1.7, 5.8, -3.0, -7.1, 1.2),
]
fig, ax = plt.subplots(figsize=(4.6, 3.1))
y = range(len(uplift))
for off, idx, color, label in (
    (-0.18, 1, BLUE, "Premium → creation"),
    (0.18, 4, RED, "Discount → redemption"),
):
    vals = [u[idx] for u in uplift]
    err = [[u[idx] - u[idx + 1] for u in uplift], [u[idx + 2] - u[idx] for u in uplift]]
    ax.barh(
        [i + off for i in y],
        vals,
        height=0.34,
        color=color,
        xerr=err,
        error_kw={"elinewidth": 0.8, "capsize": 2},
        label=label,
    )
ax.set_yticks(list(y), [u[0] for u in uplift])
ax.invert_yaxis()
ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel("Uplift vs in-band sessions (pp), 95% CI")
fig.suptitle(
    "AP response within 5 sessions of out-of-band print",
    x=0.02,
    ha="left",
    fontsize=9.5,
)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fig1_ap_response.png", dpi=220)
plt.close(fig)

# Fig 2: JNK premium vs cost band and shares outstanding, 2020.
band = pd.read_parquet(
    "data/arbitrage_band/arbitrage-20261105-v2/arbitrage_band.parquet"
)
flows = pd.read_parquet("data/premium_flows/arbitrage-20261105/flow_proxy.parquet")
lo, hi = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")
j = band[
    (band.ticker == "JNK") & band.date.between(lo, hi) & (band.status == "ok")
].sort_values("date")
s = flows[(flows.ticker == "JNK") & flows.date.between(lo, hi)].sort_values("date")
fig, (a1, a2) = plt.subplots(
    2, 1, figsize=(4.6, 3.1), sharex=True, height_ratios=[2, 1]
)
a1.fill_between(
    j.date, -j.band_bps, j.band_bps, color=GREY, alpha=0.35, lw=0, label="Cost band (±)"
)
a1.plot(j.date, j.premium_discount_bps, color="black", lw=0.7, label="Premium/discount")
for side, color in (("premium", BLUE), ("discount", RED)):
    k = j[j.side == side]
    a1.scatter(
        k.date,
        k.premium_discount_bps,
        s=5,
        color=color,
        zorder=3,
        label=f"Out of band: {side}",
    )
a1.set_ylabel("bps vs NAV")
a1.set_title(
    "JNK 2020: premium/discount, cost band, shares outstanding",
    loc="left",
    fontsize=9.5,
)
a1.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper right")
a2.plot(s.date, s.shares_outstanding / 1e6, color=BLUE, lw=1)
a2.set_ylabel("Shares (m)")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG / "fig2_jnk_2020.png", dpi=220)
plt.close(fig)

# Fig 3: next_session basket gaps over ten holdings snapshots, and the
# ex-dividend receivable that explains their day-over-day changes.
hist = pd.read_csv("data/basket_history/core-10snap/basket_history.csv")
hist = hist[hist.status == "ok"].copy()
hist["valuation_date"] = pd.to_datetime(hist["valuation_date"])
dates = sorted(hist.valuation_date.unique())
fig, (ax, bx) = plt.subplots(
    1, 2, figsize=(4.6, 3.1), gridspec_kw={"width_ratios": [1.15, 1]}
)
for ticker, d in hist.groupby("ticker"):
    x = [dates.index(v) for v in d.valuation_date]
    color = RED if ticker == "XLP" else (BLUE if ticker == "XLE" else GREY)
    ax.plot(
        x,
        d.basket_vs_nav_bps,
        color=color,
        lw=0.9 if color != GREY else 0.6,
        alpha=1 if color != GREY else 0.6,
        marker="o",
        ms=1.8,
    )
ax.axhspan(-10, 10, color=BLUE, alpha=0.08)
ax.text(1.3, -50.5, "XLP", color=RED, fontsize=6.5, va="center")
ax.text(9.2, -25.6, "XLE", color=BLUE, fontsize=6.5, va="center")
ax.set_xticks(
    range(len(dates)),
    [pd.Timestamp(v).strftime("%m-%d") for v in dates],
    rotation=90,
    fontsize=6,
)
ax.set_xlim(-0.5, 10.6)
ax.set_ylabel("Basket vs NAV (bps)", fontsize=7.5)
ax.set_xlabel("Valuation session (D+1)", fontsize=7)
ax.set_title("13 funds x 10 snapshots", loc="left", fontsize=8.5)
step = hist.dropna(subset=["gap_change_bps", "ex_dividend_bps"])
bx.scatter(step.ex_dividend_bps, step.gap_change_bps, s=7, color=BLUE, alpha=0.7)
lim = float(step.ex_dividend_bps.max()) + 1.5
bx.plot([0, lim], [0, -lim], color=RED, lw=0.8, ls="--")
bx.text(lim * 0.35, -lim * 0.9, "gap change = -ex-div", color=RED, fontsize=6)
bx.set_xlabel("Ex-dividend in interval (bps)", fontsize=7.5)
bx.set_ylabel("Gap change (bps)", fontsize=7.5)
bx.set_title("Drift = dividend receivable", loc="left", fontsize=8.5)
for a in (ax, bx):
    a.tick_params(labelsize=6.5)
fig.tight_layout()
fig.savefig(FIG / "fig3_holdings_timing.png", dpi=220)
plt.close(fig)

# Fig 4: split artifact, XLK premium before and after split correction.
before = pd.read_parquet("data/premium_flows/core-20261104/premium_discount.parquet")
after = pd.read_parquet("data/premium_flows/core-20261104-v2/premium_discount.parquet")
lo, hi = pd.Timestamp("2025-10-01"), pd.Timestamp("2026-01-31")
fig, (b1, b2) = plt.subplots(1, 2, figsize=(4.6, 2.2))
for axis, df, title, color in (
    (b1, before, "Yahoo close / NAV", RED),
    (b2, after, "Split-corrected", BLUE),
):
    d = df[(df.ticker == "XLK") & df.date.between(lo, hi)].sort_values("date")
    axis.plot(d.date, d.premium_discount_bps, color=color, lw=0.9)
    axis.axvline(pd.Timestamp("2025-12-05"), color=GREY, lw=0.6, ls="--")
    axis.set_title(title, fontsize=8.5)
    axis.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    axis.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    axis.tick_params(axis="x", labelsize=7)
b1.set_ylabel("XLK premium (bps)")
fig.suptitle(
    "XLK 2:1 split (2025-12-05): back-adjusted close vs unadjusted NAV",
    x=0.02,
    ha="left",
    fontsize=9.5,
)
fig.tight_layout()
fig.savefig(FIG / "fig4_split.png", dpi=220)
plt.close(fig)
print("figures written to", FIG)
