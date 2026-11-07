"""Shared transforms for the economic-data experiment."""

from __future__ import annotations

import pandas as pd


INDICATORS = ["gdp_growth", "inflation", "unemployment"]
INDICATOR_LABELS = {
    "gdp_growth": "GDP growth (% annual)",
    "inflation": "Inflation, CPI (% annual)",
    "unemployment": "Unemployment (% of labor force)",
}


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    wide = (
        long.pivot_table(
            index=[
                "country",
                "iso",
                "year",
                "region",
                "is_aggregate",
                "is_focus",
                "is_world",
            ],
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
