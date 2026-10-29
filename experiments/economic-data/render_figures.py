"""Save a few charts that answer the v1 question."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fetch_world_bank import CACHE_PATH, FOCUS_ISOS, load_or_fetch

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
INDICATORS = ["gdp_growth", "inflation", "unemployment"]
TITLES = {
    "gdp_growth": "GDP growth (% annual)",
    "inflation": "Inflation, CPI (% annual)",
    "unemployment": "Unemployment (% of labor force)",
}


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    wide = (
        long.pivot_table(
            index=["country", "iso", "year", "region", "is_aggregate", "is_focus", "is_world"],
            columns="indicator",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["country", "year"])
    )
    wide.columns.name = None
    return wide


def latest_year_with_coverage(countries: pd.DataFrame, col: str) -> int:
    has = countries.dropna(subset=[col])
    by_year = has.groupby("year").agg(n=("iso", "nunique"), n_focus=("is_focus", "sum"))
    eligible = by_year[(by_year["n"] >= 80) & (by_year["n_focus"] >= 6)]
    if eligible.empty:
        eligible = by_year[by_year["n"] >= 80]
    return int(eligible.index.max())


def save_focus_trends(focus: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, col in zip(axes, INDICATORS):
        sns.lineplot(data=focus, x="year", y=col, hue="country", ax=ax, legend=(col == "gdp_growth"))
        ax.set_ylabel(TITLES[col])
        ax.set_xlabel("")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    axes[-1].set_xlabel("Year")
    fig.suptitle("Six large economies, 2000–latest", y=0.99)
    fig.tight_layout()
    path = FIGURES_DIR / "focus_trends.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_world_vs_focus(wide: pd.DataFrame) -> Path:
    world = wide[wide["is_world"]]
    focus_median = (
        wide[wide["is_focus"]]
        .groupby("year")[INDICATORS]
        .median()
        .reset_index()
        .melt(id_vars="year", var_name="indicator", value_name="value")
        .assign(series="Median of six focus countries")
    )
    world_long = world.melt(
        id_vars="year", value_vars=INDICATORS, var_name="indicator", value_name="value"
    ).assign(series="World Bank World aggregate")
    plot_df = pd.concat([world_long, focus_median], ignore_index=True)
    plot_df["indicator"] = plot_df["indicator"].map(TITLES)

    g = sns.relplot(
        data=plot_df.dropna(subset=["value"]),
        x="year",
        y="value",
        hue="series",
        col="indicator",
        kind="line",
        facet_kws={"sharey": False},
        height=3.4,
        aspect=1.05,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Year", "")
    g.fig.suptitle("World aggregate vs the six-country median", y=1.05)
    path = FIGURES_DIR / "world_vs_focus.png"
    g.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    return path


def save_relationships(focus: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.scatterplot(data=focus, x="unemployment", y="inflation", hue="country", ax=axes[0])
    axes[0].set_title("Inflation vs unemployment")
    sns.scatterplot(data=focus, x="inflation", y="gdp_growth", hue="country", ax=axes[1], legend=False)
    axes[1].set_title("GDP growth vs inflation")
    fig.suptitle("Within the six-country sample")
    fig.tight_layout()
    path = FIGURES_DIR / "focus_relationships.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_global_inflation(countries: pd.DataFrame) -> Path:
    year = latest_year_with_coverage(countries, "inflation")
    snap = countries[countries["year"] == year].dropna(subset=["inflation"]).copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.histplot(data=snap, x="inflation", bins=30, ax=ax, color="#6b7c93")
    palette = sns.color_palette("tab10", n_colors=6)
    for color, (_, row) in zip(palette, snap[snap["is_focus"]].sort_values("country").iterrows()):
        ax.axvline(
            row["inflation"],
            color=color,
            linewidth=1.4,
            label=f"{row['country']} {row['inflation']:.1f}%",
        )
    ax.legend(loc="upper right", frameon=True)
    ax.set_title(f"CPI inflation across countries in {year}")
    ax.set_xlabel("Inflation, CPI (% annual)")
    fig.tight_layout()
    path = FIGURES_DIR / "global_inflation_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def render(long: pd.DataFrame | None = None) -> list[Path]:
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if long is None:
        long = load_or_fetch()
    wide = to_wide(long)
    focus = wide[wide["is_focus"]]
    countries = wide[~wide["is_aggregate"]]
    return [
        save_focus_trends(focus),
        save_world_vs_focus(wide),
        save_relationships(focus),
        save_global_inflation(countries),
    ]


def main() -> None:
    paths = render()
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
