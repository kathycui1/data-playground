from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fetch_world_bank import FOCUS_ISOS, INDICATORS, WORLD_ISO

COUNTRY_NAMES = {
    "USA": "United States",
    "CHN": "China",
    "DEU": "Germany",
    "IND": "India",
    "BRA": "Brazil",
    "JPN": "Japan",
    "WLD": "World",
}


def _value(iso: str, year: int, indicator: str) -> float:
    iso_salt = sum(ord(ch) for ch in iso) % 7
    t = year - 2000
    if indicator == "inflation":
        return 2.0 + 0.15 * iso_salt + 0.4 * np.sin(t / 3) + (0.8 if year == 2022 else 0)
    if indicator == "gdp_growth":
        return 2.5 + 0.2 * iso_salt + 0.5 * np.cos(t / 4) - (2.5 if year == 2020 else 0)
    return 5.0 + 0.3 * iso_salt - 0.05 * t


def make_panel(n_other: int = 85, years: range = range(2000, 2025)) -> pd.DataFrame:
    isos = list(FOCUS_ISOS) + [WORLD_ISO] + [f"C{i:03d}" for i in range(n_other)]
    rows = []
    for iso in isos:
        is_focus = iso in FOCUS_ISOS
        is_world = iso == WORLD_ISO
        is_aggregate = is_world
        country = COUNTRY_NAMES.get(iso, f"Country {iso}")
        for year in years:
            for code, name in INDICATORS.items():
                rows.append(
                    {
                        "country": country,
                        "iso": iso,
                        "year": year,
                        "indicator": name,
                        "indicator_code": code,
                        "value": round(_value(iso, year, name), 4),
                        "region": "East Asia & Pacific" if is_world else "Test region",
                        "income_level": "Aggregates" if is_world else "High income",
                        "is_aggregate": is_aggregate,
                        "is_focus": is_focus,
                        "is_world": is_world,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_panel()
