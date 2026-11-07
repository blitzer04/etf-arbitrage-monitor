"""Read-only Streamlit viewer for verified ETF signal evaluation bundles.

The viewer never fetches data and never computes financial values. It lists
candidate run directories, verifies the explicitly selected run through the
package, and displays the verified artifacts as stored. Charts only reshape
and filter stored values; missing values stay missing and are drawn as gaps.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from etf_monitor.bundles import (
    BundleContractError,
    BundleIncompatibleError,
    BundleIntegrityError,
    VerifiedBundle,
    compare_bundles,
    list_candidate_runs,
    verify_bundle,
)
from etf_monitor.universe import load_universe

BUNDLES_ENV = "ETF_MONITOR_BUNDLES_ROOT"
_BUNDLE_ERRORS = (BundleContractError, BundleIntegrityError, BundleIncompatibleError)
_DEFERRED = (
    (
        "Flow",
        "No qualified shares-outstanding source with 252 eligible prior "
        "intervals for all 24 ETFs (Gate 6).",
    ),
    (
        "Concentration",
        "No validated daily complete holdings for all 24 ETFs; "
        "VTI/VGT complete holdings are monthly with a ~15-day lag (Gate 7).",
    ),
)
_UNLISTED = "Not in universe config"
# Qualitative palette: category colours carry no ordering or risk meaning.
_PALETTE = px.colors.qualitative.Safe
_SIGNAL_COLOURS = {"Momentum": "#3366AA", "Volatility": "#CC7722"}
_HISTORY_WINDOWS = {"1 year": 252, "3 years": 756, "Full history": None}
_HEATMAP_WINDOWS = (21, 63, 126, 252)
_PERCENTILE_COLUMN = st.column_config.ProgressColumn(
    format="%.1f", min_value=0, max_value=100, color="#3366AA"
)


def _default_root() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bundles-root", default=None)
    known, _ = parser.parse_known_args(sys.argv[1:])
    return known.bundles_root or os.environ.get(BUNDLES_ENV, "")


@st.cache_resource(show_spinner=False, max_entries=8)
def _verified(path: str) -> VerifiedBundle:
    # Bundles are immutable once published, so a verified result is reusable
    # for the life of this process; a corrupted file is still caught on the
    # first selection because verification re-reads every byte.
    return verify_bundle(Path(path))


def _load(root: Path, run_id: str) -> VerifiedBundle | None:
    try:
        with st.spinner(f"Verifying {run_id} (hashes and full recomputation)..."):
            return _verified(str(root / run_id))
    except _BUNDLE_ERRORS as error:
        st.error(f"Run {run_id} failed verification and is not displayed: {error}")
        return None


@st.cache_data(show_spinner=False)
def _universe_labels() -> pd.DataFrame:
    universe = load_universe()
    return pd.DataFrame(
        {
            "ticker": [etf.ticker for etf in universe.etfs],
            "name": [etf.name for etf in universe.etfs],
            "category": [etf.category for etf in universe.etfs],
        }
    )


def _labels(tickers: pd.Series) -> pd.DataFrame:
    """Return ticker, name and category in universe order for the given tickers."""
    labels = _universe_labels()
    present = labels.loc[labels["ticker"].isin(tickers)]
    extra = sorted(set(tickers) - set(labels["ticker"]))
    unlisted = pd.DataFrame(
        {"ticker": extra, "name": extra, "category": [_UNLISTED] * len(extra)}
    )
    return pd.concat([present, unlisted], ignore_index=True)


def _category_order() -> dict[str, list[str]]:
    return {"category": list(_category_colours())}


def _category_colours() -> dict[str, str]:
    categories = list(dict.fromkeys(_universe_labels()["category"])) + [_UNLISTED]
    return {
        category: _PALETTE[index % len(_PALETTE)]
        for index, category in enumerate(categories)
    }


def _style(figure: go.Figure, title: str, height: int = 340) -> go.Figure:
    figure.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15}},
        template="plotly_white",
        height=height,
        margin={"t": 48, "b": 24, "l": 8, "r": 8},
        legend={"orientation": "h", "y": -0.18, "x": 0},
    )
    return figure


def _show(figure: go.Figure) -> None:
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def _value(value: object, suffix: str = "") -> str:
    if value is None or pd.isna(value):  # type: ignore[call-overload]
        return "missing"
    return f"{float(value):,.2f}{suffix}"  # type: ignore[arg-type]


def _classification_banner(bundle: VerifiedBundle) -> None:
    manifest = bundle.manifest
    if manifest["data_classification"] == "FICTIONAL":
        st.error(
            "FICTIONAL synthetic data. These values are not market results and "
            "must not be interpreted as such."
        )
    else:
        st.info(
            "Observed provider data (Yahoo chart v8, undocumented endpoint, local "
            "research only). History is a current-vintage reconstruction, not a "
            "point-in-time backtest."
        )
    st.caption(
        f"Run {bundle.run_id} | target session {manifest['target_session']} | "
        f"evaluation instant {manifest['evaluation_instant_utc']} | "
        f"sources {', '.join(map(str, manifest['sources']))}"  # type: ignore[call-overload]
    )


def _run_summary(bundle: VerifiedBundle) -> None:
    coverage = bundle.frame("coverage")
    count = len(coverage)
    staleness = coverage["price_staleness_sessions"].max()
    columns = st.columns(5)
    columns[0].metric("Target session", str(bundle.manifest["target_session"]))
    columns[1].metric("ETFs in run", count)
    columns[2].metric(
        "Momentum eligible",
        f"{coverage['momentum_target_status'].eq('eligible').sum()} / {count}",
        help="ETFs with an eligible 252-session return on the target session.",
    )
    columns[3].metric(
        "Volatility eligible",
        f"{coverage['volatility_target_status'].eq('eligible').sum()} / {count}",
        help="ETFs with an eligible 21-return volatility on the target session.",
    )
    columns[4].metric(
        "Max price staleness",
        "missing" if pd.isna(staleness) else f"{int(staleness)} sessions",
        help="Sessions between the target session and the last eligible "
        "adjusted close, worst ETF.",
    )


def _ranking_bar(
    table: pd.DataFrame, value: str, percentile: str, title: str, axis: str
) -> go.Figure:
    present = table.loc[table[value].notna()].sort_values(value)
    figure = px.bar(
        present,
        x=value,
        y="ticker",
        orientation="h",
        color="category",
        color_discrete_map=_category_colours(),
        category_orders=_category_order(),
        custom_data=["name", percentile],
    )
    figure.update_traces(
        hovertemplate="<b>%{y}</b> %{customdata[0]}<br>"
        + axis
        + ": %{x:.2f}<br>Own-history percentile: %{customdata[1]:.1f}"
        "<extra></extra>"
    )
    figure.update_layout(
        xaxis_title=axis,
        yaxis_title=None,
        yaxis={"categoryorder": "array", "categoryarray": list(present["ticker"])},
        legend_title_text=None,
    )
    return _style(figure, title, height=max(320, 22 * len(present) + 120))


def _percentile_scatter(table: pd.DataFrame) -> go.Figure:
    present = table.dropna(subset=["momentum_pct", "volatility_pct"])
    figure = px.scatter(
        present,
        x="momentum_pct",
        y="volatility_pct",
        text="ticker",
        color="category",
        color_discrete_map=_category_colours(),
        category_orders=_category_order(),
        custom_data=["name"],
    )
    figure.update_traces(
        textposition="top center",
        marker={"size": 11, "line": {"width": 1, "color": "white"}},
        hovertemplate="<b>%{text}</b> %{customdata[0]}<br>Momentum percentile: "
        "%{x:.1f}<br>Volatility percentile: %{y:.1f}<extra></extra>",
    )
    for axis in ("x", "y"):
        figure.add_shape(
            type="line",
            line={"dash": "dot", "color": "#999999", "width": 1},
            x0=50 if axis == "x" else 0,
            x1=50 if axis == "x" else 100,
            y0=0 if axis == "x" else 50,
            y1=100 if axis == "x" else 50,
        )
    figure.update_layout(
        xaxis={"title": "Momentum percentile (own history)", "range": [-2, 102]},
        yaxis={"title": "Volatility percentile (own history)", "range": [-2, 102]},
        legend_title_text=None,
    )
    return _style(figure, "Target-session percentiles", height=480)


def _percentile_heatmap(
    signal: pd.DataFrame, order: list[str], sessions: int, label: str
) -> go.Figure:
    dates = sorted(signal["signal_date"].unique())[-sessions:]
    recent = signal.loc[signal["signal_date"].isin(dates)]
    grid = recent.pivot(index="ticker", columns="signal_date", values="percentile")
    grid = grid.reindex(index=[ticker for ticker in order if ticker in grid.index])
    figure = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(dtype=float),
            x=list(grid.columns),
            y=list(grid.index),
            zmin=0,
            zmax=100,
            colorscale="Cividis",
            colorbar={"title": "pctile"},
            hoverongaps=False,
            hovertemplate="%{y} %{x|%Y-%m-%d}<br>percentile %{z:.1f}<extra></extra>",
        )
    )
    figure.update_layout(yaxis={"autorange": "reversed"}, xaxis_title=None)
    return _style(
        figure,
        f"{label} percentile, last {sessions} sessions (blank = not available)",
        height=max(320, 20 * len(grid) + 140),
    )


def _overview(bundle: VerifiedBundle) -> None:
    coverage = bundle.frame("coverage")
    labels = _labels(coverage["ticker"])
    table = labels.merge(
        pd.DataFrame(
            {
                "ticker": coverage["ticker"],
                "momentum_return": coverage["momentum_target_display"],
                "momentum_pct": coverage["momentum_target_percentile"],
                "momentum_status": coverage["momentum_target_status"],
                "momentum_norm": coverage["momentum_target_normalization_status"],
                "volatility_value": coverage["volatility_target_display"],
                "volatility_pct": coverage["volatility_target_percentile"],
                "volatility_status": coverage["volatility_target_status"],
                "volatility_norm": coverage["volatility_target_normalization_status"],
                "staleness": coverage["price_staleness_sessions"],
            }
        ),
        on="ticker",
        how="left",
        validate="one_to_one",
    )
    st.subheader(f"Target session {bundle.manifest['target_session']}")
    st.write(
        "Standalone diagnostics only. Each percentile is that ETF's own trailing "
        "756-session midrank (current value excluded), so percentiles describe "
        "an ETF relative to its own history, not relative to other ETFs. There "
        "is no composite score and no risk classification. A value missing on "
        "the target session is shown as missing; it is never replaced by an "
        "earlier value."
    )

    left, right = st.columns(2)
    with left:
        _show(
            _ranking_bar(
                table,
                "momentum_return",
                "momentum_pct",
                "Momentum: 252-session return",
                "Return (%)",
            )
        )
    with right:
        _show(
            _ranking_bar(
                table,
                "volatility_value",
                "volatility_pct",
                "Volatility: 21-return annualized",
                "Annualized volatility (%)",
            )
        )
    for column, name in (
        ("momentum_return", "Momentum"),
        ("volatility_value", "Volatility"),
    ):
        missing = table.loc[table[column].isna(), "ticker"]
        if not missing.empty:
            st.caption(
                f"{name} not available on the target session: {', '.join(missing)}."
            )

    _show(_percentile_scatter(table))
    st.caption(
        "Dotted lines mark each ETF's own-history median (50). Regions of this "
        "chart are not labelled or scored; the two percentiles are separate "
        "diagnostics."
    )

    st.subheader("Recent percentile history")
    signal_label = st.radio(
        "Signal", ["Momentum", "Volatility"], horizontal=True, key="heatmap_signal"
    )
    sessions = st.select_slider(
        "Sessions shown", options=_HEATMAP_WINDOWS, value=63, key="heatmap_sessions"
    )
    signal = bundle.frame("momentum" if signal_label == "Momentum" else "volatility")
    _show(
        _percentile_heatmap(signal, list(labels["ticker"]), int(sessions), signal_label)
    )

    st.subheader("Target-session values")
    st.dataframe(
        table.rename(
            columns={
                "ticker": "ETF",
                "name": "Name",
                "category": "Category",
                "momentum_return": "252d return (%)",
                "momentum_pct": "Momentum pctile",
                "momentum_status": "Momentum status",
                "momentum_norm": "Momentum norm. status",
                "volatility_value": "21d ann. vol (%)",
                "volatility_pct": "Volatility pctile",
                "volatility_status": "Volatility status",
                "volatility_norm": "Volatility norm. status",
                "staleness": "Price staleness (sessions)",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "252d return (%)": st.column_config.NumberColumn(format="%.2f"),
            "21d ann. vol (%)": st.column_config.NumberColumn(format="%.2f"),
            "Momentum pctile": _PERCENTILE_COLUMN,
            "Volatility pctile": _PERCENTILE_COLUMN,
        },
    )

    st.subheader("Coverage")
    st.dataframe(
        coverage[
            [
                "ticker",
                "expected_session_count",
                "present_row_count",
                "missing_row_count",
                "missing_adjusted_close_count",
                "rows_after_target",
                "first_row_date",
                "last_row_date",
                "last_eligible_adjusted_close_date",
                "retrieved_at_min",
                "retrieved_at_max",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    st.subheader("Not calculated")
    for name, reason in _DEFERRED:
        st.markdown(f"**{name}: deferred, not calculated.** {reason}")


def _history_line(
    frame: pd.DataFrame,
    columns: dict[str, str],
    title: str,
    y_title: str,
    *,
    log_y: bool = False,
    reference: float | None = None,
    y_range: list[float] | None = None,
) -> go.Figure:
    figure = go.Figure()
    for column, label in columns.items():
        figure.add_trace(
            go.Scatter(
                x=frame["signal_date"],
                y=frame[column],
                mode="lines",
                connectgaps=False,
                name=label,
                line={"width": 1.6, "color": _SIGNAL_COLOURS.get(label)},
                hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra>" + label + "</extra>",
            )
        )
    if reference is not None:
        figure.add_hline(y=reference, line={"dash": "dot", "color": "#999999"})
    figure.update_layout(
        yaxis={"title": y_title, "type": "log" if log_y else "linear"},
        hovermode="x unified",
        showlegend=len(columns) > 1,
    )
    if y_range is not None:
        figure.update_yaxes(range=y_range)
    return _style(figure, title)


def _etf_detail(bundle: VerifiedBundle) -> None:
    coverage = bundle.frame("coverage")
    labels = _labels(coverage["ticker"]).set_index("ticker")
    ticker = st.selectbox(
        "ETF",
        list(labels.index),
        key="detail_ticker",
        format_func=lambda symbol: f"{symbol} - {labels.at[symbol, 'name']}",
    )
    row = coverage.loc[coverage["ticker"].eq(ticker)].iloc[0]
    momentum = bundle.frame("momentum")
    volatility = bundle.frame("volatility")
    momentum = momentum.loc[momentum["ticker"].eq(ticker)]
    volatility = volatility.loc[volatility["ticker"].eq(ticker)]

    st.caption(f"Category: {labels.at[ticker, 'category']}")
    cards = st.columns(4)
    cards[0].metric(
        "252d return",
        _value(row["momentum_target_display"], "%"),
        help=f"Status: {row['momentum_target_status']}",
    )
    cards[1].metric(
        "Momentum percentile",
        _value(row["momentum_target_percentile"]),
        help=f"Normalization: {row['momentum_target_normalization_status']}; "
        f"{row['momentum_target_reference_count']} reference sessions",
    )
    cards[2].metric(
        "21d annualized vol",
        _value(row["volatility_target_display"], "%"),
        help=f"Status: {row['volatility_target_status']}",
    )
    cards[3].metric(
        "Volatility percentile",
        _value(row["volatility_target_percentile"]),
        help=f"Normalization: {row['volatility_target_normalization_status']}; "
        f"{row['volatility_target_reference_count']} reference sessions",
    )

    window = st.radio(
        "History shown",
        list(_HISTORY_WINDOWS),
        index=1,
        horizontal=True,
        key="detail_window",
    )
    sessions = _HISTORY_WINDOWS[window]
    if sessions is not None:
        momentum_view = momentum.tail(sessions)
        volatility_view = volatility.tail(sessions)
    else:
        momentum_view, volatility_view = momentum, volatility
    st.caption(
        "Signals are labelled as of the close of signal_date; first_use_session is "
        "the next XNYS session. Gaps are drawn as gaps."
    )
    left, right = st.columns(2)
    with left:
        _show(
            _history_line(
                momentum_view,
                {"end_adjusted_close": "Adjusted close"},
                f"{ticker} adjusted close (log scale)",
                "Adjusted close",
                log_y=True,
            )
        )
        _show(
            _history_line(
                momentum_view,
                {"simple_return_pct": "Momentum"},
                f"{ticker} 252-session return",
                "%",
                reference=0.0,
            )
        )
    with right:
        _show(
            _history_line(
                volatility_view,
                {"display_vol_pct": "Volatility"},
                f"{ticker} 21-return annualized volatility",
                "%",
            )
        )
        percentiles = momentum_view[["signal_date"]].assign(
            momentum=momentum_view["percentile"].to_numpy()
        )
        percentiles = percentiles.merge(
            volatility_view[["signal_date", "percentile"]].rename(
                columns={"percentile": "volatility"}
            ),
            on="signal_date",
            how="outer",
            validate="one_to_one",
        )
        _show(
            _history_line(
                percentiles,
                {"momentum": "Momentum", "volatility": "Volatility"},
                f"{ticker} own-history percentiles",
                "pctile",
                reference=50.0,
                y_range=[0, 100],
            )
        )
    counts = pd.concat(
        {
            "momentum": momentum["status"].value_counts(),
            "momentum normalization": momentum["normalization_status"].value_counts(),
            "volatility": volatility["status"].value_counts(),
            "volatility normalization": volatility[
                "normalization_status"
            ].value_counts(),
        }
    ).rename("sessions")
    st.subheader("Status counts")
    st.dataframe(counts.reset_index(), hide_index=True)
    with st.expander("Momentum rows"):
        st.dataframe(momentum, hide_index=True, width="stretch")
    with st.expander("Volatility rows"):
        st.dataframe(volatility, hide_index=True, width="stretch")


def _dependence(bundle: VerifiedBundle) -> None:
    st.write(
        "Descriptive correlation between Momentum and Volatility percentiles. "
        "No p-values, no pooled estimate, no causal or predictive claim. "
        "Sessions with fewer than 3 pairs or constant input are reported by "
        "status, not as a number."
    )
    cross = bundle.frame("dependence_cross_section")
    figure = _history_line(
        cross,
        {"pearson": "Pearson", "spearman": "Spearman"},
        "Cross-sectional correlation by session",
        "correlation",
        reference=0.0,
        y_range=[-1, 1],
    )
    figure.update_traces(selector={"name": "Pearson"}, line={"color": "#3366AA"})
    figure.update_traces(selector={"name": "Spearman"}, line={"color": "#CC7722"})
    _show(figure)
    st.dataframe(
        cross["status"].value_counts().rename("sessions").reset_index(),
        hide_index=True,
    )

    st.subheader("Per-ETF correlation over full history")
    per_etf = bundle.frame("dependence_per_etf")
    numeric = per_etf.dropna(subset=["pearson", "spearman"], how="all")
    if not numeric.empty:
        long = numeric.melt(
            id_vars=["ticker"],
            value_vars=["pearson", "spearman"],
            var_name="measure",
            value_name="correlation",
        )
        bars = px.bar(
            long,
            x="ticker",
            y="correlation",
            color="measure",
            barmode="group",
            color_discrete_map={"pearson": "#3366AA", "spearman": "#CC7722"},
            category_orders={"ticker": list(per_etf["ticker"])},
        )
        bars.update_layout(
            yaxis_range=[-1, 1], xaxis_title=None, legend_title_text=None
        )
        _show(_style(bars, "Per-ETF time-series correlation of the two percentiles"))
    st.dataframe(per_etf, hide_index=True, width="stretch")


def _provenance(bundle: VerifiedBundle) -> None:
    manifest = bundle.manifest
    st.subheader("Limitations")
    for item in manifest["limitations"]:  # type: ignore[attr-defined]
        st.markdown(f"- {item}")
    for name, reason in _DEFERRED:
        st.markdown(f"- {name}: deferred, not calculated. {reason}")
    st.subheader("Manifest")
    st.json(dict(manifest), expanded=False)


def _comparison(root: Path, bundle: VerifiedBundle, runs: list[str]) -> None:
    others = [run for run in runs if run != bundle.run_id]
    other_id = st.selectbox(
        "Compare against run",
        others,
        index=None,
        placeholder="Select a second run",
        key=f"compare_{bundle.run_id}",
    )
    if other_id is None:
        st.write("No comparison run selected.")
        return
    other = _load(root, other_id)
    if other is None:
        return
    try:
        comparison = compare_bundles(bundle, other)
    except BundleIncompatibleError as error:
        st.error(f"Runs are not comparable: {error}")
        return
    st.write(
        f"A = {bundle.run_id}, B = {other.run_id}. Counts are descriptive; the "
        "cause of a provider revision is not inferred."
    )
    counts = comparison.input_row_counts.iloc[0]
    target = comparison.target_session_changes
    flags = target.select_dtypes(include="bool")
    cards = st.columns(4)
    cards[0].metric("Overlapping input rows", f"{int(counts['overlapping_rows']):,}")
    cards[1].metric(
        "Rows only in A / only in B",
        f"{int(counts['rows_only_in_a']):,} / {int(counts['rows_only_in_b']):,}",
    )
    fields = comparison.input_field_changes
    value_fields = fields.loc[~fields["field"].isin(["retrieved_at", "source"])]
    cards[2].metric(
        "Changed price/volume values",
        f"{int(value_fields['changed_count'].sum()):,}",
        help="Excludes retrieved_at and source metadata, which differ by design.",
    )
    cards[3].metric(
        "Target-session signal rows changed",
        f"{int(flags.any(axis=1).sum())} / {len(target)}",
    )

    changes = comparison.signal_changes_by_ticker.melt(
        id_vars=["ticker", "signal"],
        value_vars=["raw_changed_count", "percentile_changed_count"],
        var_name="measure",
        value_name="sessions",
    )
    changes["measure"] = changes["measure"].map(
        {
            "raw_changed_count": "raw value changed",
            "percentile_changed_count": "percentile changed",
        }
    )
    figure = px.bar(
        changes,
        x="ticker",
        y="sessions",
        color="signal",
        barmode="group",
        facet_row="measure",
        color_discrete_map={"momentum": "#3366AA", "volatility": "#CC7722"},
    )
    figure.update_yaxes(matches=None, title=None)
    figure.for_each_annotation(lambda note: note.update(text=note.text.split("=")[-1]))
    figure.update_layout(xaxis_title=None, legend_title_text=None)
    _show(
        _style(figure, "Historical sessions whose signal changed, by ETF", height=520)
    )

    for title, frame in (
        ("Input row counts", comparison.input_row_counts),
        ("Input field changes", comparison.input_field_changes),
        ("Input changes by ETF", comparison.input_changes_by_ticker),
        ("Signal changes by ETF", comparison.signal_changes_by_ticker),
        ("Target-session changes", comparison.target_session_changes),
    ):
        st.subheader(title)
        st.dataframe(frame, hide_index=True, width="stretch")


def main() -> None:
    """Render the viewer for one explicitly selected verified run."""
    st.set_page_config(page_title="ETF signal monitor (read-only)", layout="wide")
    st.title("U.S. ETF standalone price diagnostics")
    st.caption(
        "Momentum (252-session return) and Volatility (21-return annualized) for "
        "24 U.S. equity ETFs, each ranked against its own trailing history. "
        "Read-only view of verified run bundles."
    )
    root_text = st.sidebar.text_input("Bundles root directory", _default_root())
    if not root_text:
        st.info(f"Set a bundles root (--bundles-root or {BUNDLES_ENV}).")
        return
    root = Path(root_text).expanduser()
    try:
        runs = list_candidate_runs(root)
    except BundleContractError as error:
        st.error(str(error))
        return
    run_id = st.sidebar.selectbox(
        "Run", runs, index=None, placeholder="Select a run", key="run_id"
    )
    st.sidebar.caption(
        "No run is selected by default. Candidates are unverified until selected."
    )
    if run_id is None:
        st.info("Select a run in the sidebar. Nothing is shown until it verifies.")
        return
    bundle = _load(root, run_id)
    if bundle is None:
        return
    _classification_banner(bundle)
    _run_summary(bundle)
    overview, detail, dependence, comparison, provenance = st.tabs(
        [
            "Overview",
            "ETF detail",
            "Dependence",
            "Vintage comparison",
            "Provenance & limitations",
        ]
    )
    with overview:
        _overview(bundle)
    with detail:
        _etf_detail(bundle)
    with dependence:
        _dependence(bundle)
    with comparison:
        _comparison(root, bundle, runs)
    with provenance:
        _provenance(bundle)


main()
