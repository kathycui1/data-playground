"""Build a single HTML report with charts and summary tables."""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from plotly.subplots import make_subplots

from fetch_world_bank import CACHE_PATH, load_or_fetch
from panels import INDICATORS, latest_year_with_coverage, to_wide

REPORT_PATH = Path(__file__).resolve().parent / "report.html"

COUNTRY_COLORS = {
    "Brazil": "#1f77b4",
    "China": "#d62728",
    "Germany": "#2ca02c",
    "India": "#9467bd",
    "Japan": "#ff7f0e",
    "United States": "#17becf",
}
SHORT_TITLES = {
    "gdp_growth": "GDP growth",
    "inflation": "Inflation (CPI)",
    "unemployment": "Unemployment",
}
AXIS_LABELS = {
    "gdp_growth": "GDP %",
    "inflation": "Inflation %",
    "unemployment": "Unemp. %",
}
TERM_LABELS = {
    "Intercept": "Intercept (Brazil baseline)",
    "unemployment": "Unemployment",
}

STYLES = """
:root {
  color-scheme: light dark;
  --bg: #f4f6f8;
  --card: #ffffff;
  --text: #1b1f24;
  --muted: #5b6570;
  --border: #d5dce6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #121418;
    --card: #1c2128;
    --text: #e6edf3;
    --muted: #9aa5b1;
    --border: #30363d;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 2rem 1.25rem 3rem;
  font-family: "Segoe UI", system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
}
main { max-width: 72rem; margin: 0 auto; }
h1 { font-size: 1.65rem; font-weight: 650; margin: 0 0 0.4rem; letter-spacing: -0.02em; }
.lead { color: var(--muted); margin: 0 0 1.5rem; max-width: 48rem; }
section {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.25rem 1.35rem;
  margin-bottom: 1.15rem;
}
h2 { font-size: 1.05rem; font-weight: 650; margin: 0 0 0.35rem; }
.blurb { color: var(--muted); font-size: 0.9rem; margin: 0 0 0.85rem; }
.pipeline {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.45rem 0.2rem;
  font-size: 0.9rem;
}
.pipe-step {
  padding: 0.4rem 0.7rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
}
.pipe-arrow { color: var(--muted); padding: 0 0.15rem; }
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
th, td { text-align: left; padding: 0.45rem 0.55rem; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 550; }
td.num { font-variant-numeric: tabular-nums; }
.caveats { font-size: 0.9rem; color: var(--muted); margin: 0; }
.meta { font-size: 0.85rem; color: var(--muted); margin: 0.35rem 0 0.75rem; }
.plot { margin-top: 0.25rem; width: 100%; }
.focus-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin: 0 0 0.85rem;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.3rem 0.65rem;
  border: 1px solid var(--border);
  border-radius: 999px;
  font-size: 0.85rem;
  background: var(--bg);
}
.chip-swatch {
  width: 0.7rem;
  height: 0.7rem;
  border-radius: 50%;
  flex: 0 0 auto;
}
"""


def _base_layout(**kwargs) -> dict:
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, system-ui, sans-serif", size=13, color="#1b1f24"),
        hoverlabel=dict(bgcolor="white", font_size=12, namelength=-1),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        margin=dict(l=58, r=24, t=36, b=88),
    )
    layout.update(kwargs)
    return layout


def _style_axes(fig: go.Figure) -> None:
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e8edf2",
        zeroline=False,
        showline=False,
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e8edf2",
        zeroline=False,
        showline=False,
        automargin=True,
    )


def _world_vs_focus_figure(wide: pd.DataFrame) -> go.Figure:
    world = wide[wide["is_world"]]
    focus_median = (
        wide[wide["is_focus"]].groupby("year")[INDICATORS].median().reset_index()
    )
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[SHORT_TITLES[c] for c in INDICATORS],
        horizontal_spacing=0.07,
    )
    for col_idx, indicator in enumerate(INDICATORS, start=1):
        w = world.dropna(subset=[indicator])
        fig.add_trace(
            go.Scatter(
                x=w["year"],
                y=w[indicator],
                name="World",
                mode="lines",
                line=dict(color="#245b8a", width=2.6),
                legendgroup="world",
                showlegend=col_idx == 1,
                hovertemplate="World<br>%{x}: %{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                x=focus_median["year"],
                y=focus_median[indicator],
                name="Six-country median",
                mode="lines",
                line=dict(color="#c05621", width=2.6, dash="dot"),
                legendgroup="median",
                showlegend=col_idx == 1,
                hovertemplate="Six-country median<br>%{x}: %{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(title_text="%", row=1, col=col_idx)
        fig.update_xaxes(title_text="Year", row=1, col=col_idx)
    fig.update_layout(
        **_base_layout(
            height=390,
            hovermode="x unified",
            margin=dict(l=50, r=18, t=48, b=96),
        )
    )
    fig.update_annotations(font_size=13, yshift=8)
    _style_axes(fig)
    return fig


def _focus_trends_figure(focus: pd.DataFrame) -> go.Figure:
    # One chart per indicator, titles as y-axis labels so they never collide with legends.
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
    )
    for row_idx, indicator in enumerate(INDICATORS, start=1):
        for country in sorted(focus["country"].unique()):
            sub = focus[focus["country"] == country].dropna(subset=[indicator])
            fig.add_trace(
                go.Scatter(
                    x=sub["year"],
                    y=sub[indicator],
                    name=country,
                    legendgroup=country,
                    showlegend=False,
                    mode="lines",
                    line=dict(color=COUNTRY_COLORS[country], width=2.3),
                    hovertemplate=f"{country}<br>%{{x}}: %{{y:.2f}}%<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )
        fig.update_yaxes(title_text=AXIS_LABELS[indicator], row=row_idx, col=1)
    fig.update_xaxes(title_text="Year", row=3, col=1)
    fig.update_layout(
        **_base_layout(
            height=780,
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=58, r=20, t=16, b=48),
        )
    )
    _style_axes(fig)
    return fig


def _relationships_figure(focus: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Inflation vs unemployment", "GDP growth vs inflation"],
        horizontal_spacing=0.1,
    )
    for country in sorted(focus["country"].unique()):
        sub = focus[focus["country"] == country]
        color = COUNTRY_COLORS[country]
        left = sub.dropna(subset=["unemployment", "inflation"])
        fig.add_trace(
            go.Scatter(
                x=left["unemployment"],
                y=left["inflation"],
                name=country,
                legendgroup=country,
                showlegend=False,
                mode="markers",
                marker=dict(color=color, size=8, opacity=0.78, line=dict(width=0)),
                hovertemplate=(
                    f"{country}<br>Unemployment: %{{x:.1f}}%<br>"
                    "Inflation: %{y:.1f}%<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        right = sub.dropna(subset=["inflation", "gdp_growth"])
        fig.add_trace(
            go.Scatter(
                x=right["inflation"],
                y=right["gdp_growth"],
                name=country,
                legendgroup=country,
                showlegend=False,
                mode="markers",
                marker=dict(color=color, size=8, opacity=0.78, line=dict(width=0)),
                hovertemplate=(
                    f"{country}<br>Inflation: %{{x:.1f}}%<br>"
                    "GDP growth: %{y:.1f}%<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text=AXIS_LABELS["unemployment"], row=1, col=1)
    fig.update_yaxes(title_text=AXIS_LABELS["inflation"], row=1, col=1)
    fig.update_xaxes(title_text=AXIS_LABELS["inflation"], row=1, col=2)
    fig.update_yaxes(title_text=AXIS_LABELS["gdp_growth"], row=1, col=2)
    fig.update_layout(
        **_base_layout(
            height=400,
            showlegend=False,
            margin=dict(l=60, r=20, t=48, b=56),
        )
    )
    fig.update_annotations(font_size=13, yshift=8)
    _style_axes(fig)
    return fig


def _inflation_histogram(countries: pd.DataFrame):
    year = latest_year_with_coverage(countries, "inflation")
    snap = countries[countries["year"] == year].dropna(subset=["inflation"]).copy()
    # Keep the bulk of the distribution readable; extreme outliers flatten the chart.
    xmax = 15.0
    clipped = snap[snap["inflation"].between(-5, xmax)]
    n_hidden = int((~snap["inflation"].between(-5, xmax)).sum())
    focus_rows = (
        snap[snap["is_focus"]]
        .sort_values("country")[["country", "inflation"]]
        .reset_index(drop=True)
    )

    fig = go.Figure(
        data=[
            go.Histogram(
                x=clipped["inflation"],
                xbins=dict(start=-5, end=xmax, size=1),
                marker=dict(color="#7b8ea3", line=dict(color="#ffffff", width=0.5)),
                hovertemplate="Countries in bin: %{y}<extra></extra>",
                showlegend=False,
            )
        ]
    )
    # Subtle markers at focus values, without text annotations that collide.
    for _, row in focus_rows.iterrows():
        if -5 <= row["inflation"] <= xmax:
            fig.add_vline(
                x=row["inflation"],
                line=dict(color=COUNTRY_COLORS[row["country"]], width=2.2),
                opacity=0.9,
            )

    title = f"CPI inflation across countries in {year}"
    if n_hidden:
        title += f" ({n_hidden} countries outside −5% to {xmax:.0f}% not shown)"

    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=14), x=0, xanchor="left"),
            height=380,
            bargap=0.04,
            showlegend=False,
            margin=dict(l=60, r=24, t=56, b=56),
            xaxis_title="Inflation, CPI (% annual)",
            yaxis_title="Number of countries",
            xaxis=dict(range=[-5, xmax]),
        )
    )
    _style_axes(fig)
    return fig, year, focus_rows, n_hidden, xmax


def _country_legend_html(values: dict[str, str] | None = None) -> str:
    chips = []
    for country, color in COUNTRY_COLORS.items():
        label = html.escape(country)
        if values and country in values:
            label = f"{label} <strong>{html.escape(values[country])}</strong>"
        chips.append(
            '<span class="chip">'
            f'<span class="chip-swatch" style="background:{color}"></span>'
            f"{label}"
            "</span>"
        )
    return '<div class="focus-chips">' + "".join(chips) + "</div>"


def _correlation_table(focus: pd.DataFrame) -> pd.DataFrame:
    corr = focus[INDICATORS].corr().round(3)
    corr.index = [SHORT_TITLES[i] for i in corr.index]
    corr.columns = [SHORT_TITLES[c] for c in corr.columns]
    return corr.reset_index(names="Indicator")


def _pretty_term(term: str) -> str:
    if term in TERM_LABELS:
        return TERM_LABELS[term]
    match = re.fullmatch(r"C\(country\)\[T\.(.+)\]", term)
    if match:
        return f"{match.group(1)} (vs Brazil)"
    return term


def _regression_table(focus: pd.DataFrame):
    model_df = focus.dropna(subset=["inflation", "unemployment"])
    model = smf.ols("inflation ~ unemployment + C(country)", data=model_df).fit()
    summary = pd.DataFrame(
        {
            "Term": [_pretty_term(t) for t in model.params.index],
            "Coefficient": model.params.values.round(3),
            "Std. error": model.bse.values.round(3),
            "P-value": model.pvalues.values.round(3),
        }
    )
    return summary, model


def _df_to_html_table(frame: pd.DataFrame) -> str:
    numeric_cols = {
        c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])
    }
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for col in frame.columns:
            value = row[col]
            text = html.escape(str(value))
            css = ' class="num"' if col in numeric_cols else ""
            cells.append(f"<td{css}>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table><thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def build_report(long: pd.DataFrame | None = None, path: Path = REPORT_PATH) -> Path:
    if long is None:
        long = load_or_fetch(CACHE_PATH)

    wide = to_wide(long)
    focus = wide[wide["is_focus"]].copy()
    countries = wide[~wide["is_aggregate"]].copy()
    n_countries = int(countries["iso"].nunique())
    year_min, year_max = int(wide["year"].min()), int(wide["year"].max())

    corr_html = _df_to_html_table(_correlation_table(focus))
    reg_table, model = _regression_table(focus)
    hist_fig, snap_year, focus_rows, n_hidden, xmax = _inflation_histogram(countries)

    inflation_values = {
        row["country"]: f"{row['inflation']:.1f}%" for _, row in focus_rows.iterrows()
    }
    figures = [
        (
            "World vs six-country median",
            "World Bank World aggregate compared with the median of the six large economies.",
            _world_vs_focus_figure(wide),
            "",
        ),
        (
            "Six large economies over time",
            "GDP growth, inflation, and unemployment for Brazil, China, Germany, India, Japan, and the United States.",
            _focus_trends_figure(focus),
            _country_legend_html(),
        ),
        (
            "How the indicators relate",
            "Each point is one country-year in the six-country sample.",
            _relationships_figure(focus),
            _country_legend_html(),
        ),
        (
            "Global inflation snapshot",
            "Where the six economies sit in the cross-country inflation distribution.",
            hist_fig,
            _country_legend_html(inflation_values),
        ),
    ]

    plot_fragments = []
    for title, blurb, fig, extra in figures:
        fragment = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"displayModeBar": False, "responsive": True},
        )
        plot_fragments.append(
            "<section>"
            f"<h2>{html.escape(title)}</h2>"
            f'<p class="blurb">{html.escape(blurb)}</p>'
            f"{extra}"
            f'<div class="plot">{fragment}</div>'
            "</section>"
        )

    plotly_cdn = (
        '<script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    )
    outlier_note = (
        f" Inflation chart hides {n_hidden} countries outside −5% to {xmax:.0f}%."
        if n_hidden
        else ""
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Economic indicators — World Bank</title>
  <style>{STYLES}</style>
  {plotly_cdn}
</head>
<body>
  <main>
    <h1>Economic indicators</h1>
    <p class="lead">
      Since {year_min}, how do GDP growth, inflation, and unemployment move together
      globally and in six large economies? Source: World Bank ({n_countries} countries;
      years through {year_max}).
    </p>

    <section>
      <h2>Data flow</h2>
      <div class="pipeline" role="img" aria-label="World Bank API to parquet cache to analysis to HTML report">
        <span class="pipe-step">World Bank API</span>
        <span class="pipe-arrow" aria-hidden="true">→</span>
        <span class="pipe-step">Parquet cache</span>
        <span class="pipe-arrow" aria-hidden="true">→</span>
        <span class="pipe-step">Analysis</span>
        <span class="pipe-arrow" aria-hidden="true">→</span>
        <span class="pipe-step">This report</span>
      </div>
    </section>

    {"".join(plot_fragments)}

    <section>
      <h2>Correlations (six-country panel)</h2>
      <p class="blurb">Pooled country-year correlations. Pairwise complete observations.</p>
      {corr_html}
    </section>

    <section>
      <h2>Regression: inflation on unemployment</h2>
      <p class="meta">
        Pooled OLS with country intercepts on the six-country sample.
        R² = {model.rsquared:.3f}, n = {int(model.nobs)}. Descriptive association only.
      </p>
      {_df_to_html_table(reg_table)}
    </section>

    <section>
      <h2>Notes</h2>
      <p class="caveats">
        Annual World Bank series can lag and omit country-years (especially unemployment).
        The World aggregate is size-weighted; the six-country median is not.
        Correlation and regression coefficients are not causal claims.
        Inflation snapshot year: {snap_year}.{outlier_note}
      </p>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
    return path


def main() -> None:
    path = build_report()
    print(path)


if __name__ == "__main__":
    main()
