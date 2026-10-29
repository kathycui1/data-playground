"""Fetch World Bank indicators for all countries and cache them as parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.worldbank.org/v2"
FOCUS_ISOS = ("USA", "CHN", "DEU", "IND", "BRA", "JPN")
WORLD_ISO = "WLD"
INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG": "inflation",
    "SL.UEM.TOTL.ZS": "unemployment",
}
START_YEAR = 2000
END_YEAR = 2026
PER_PAGE = 20000
MIN_COUNTRIES = 50

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
CACHE_PATH = DATA_DIR / "world_bank_indicators.parquet"


def _get_json(url: str, params: dict) -> list:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Unexpected response from {url}")
    return payload


def fetch_country_meta() -> pd.DataFrame:
    url = f"{BASE_URL}/country"
    page = 1
    records: list[dict] = []
    while True:
        payload = _get_json(
            url, {"format": "json", "per_page": 400, "page": page}
        )
        meta, rows = payload[0], payload[1] or []
        records.extend(rows)
        if page >= int(meta.get("pages", 1)):
            break
        page += 1

    return pd.DataFrame(
        [
            {
                "iso": row.get("id"),
                "country": row.get("name"),
                "region": (row.get("region") or {}).get("value"),
                "income_level": (row.get("incomeLevel") or {}).get("value"),
                "is_aggregate": (row.get("region") or {}).get("value") == "Aggregates",
            }
            for row in records
            if row.get("id")
        ]
    )


def fetch_indicator(indicator_code: str, countries: str) -> list[dict]:
    url = f"{BASE_URL}/country/{countries}/indicator/{indicator_code}"
    page = 1
    records: list[dict] = []
    while True:
        payload = _get_json(
            url,
            {
                "format": "json",
                "date": f"{START_YEAR}:{END_YEAR}",
                "per_page": PER_PAGE,
                "page": page,
            },
        )
        if len(payload) < 2 or payload[1] is None:
            raise RuntimeError(f"No data returned for indicator {indicator_code}")
        records.extend(payload[1])
        if page >= int(payload[0].get("pages", 1)):
            break
        page += 1
    return records


def rows_from_payload(indicator_code: str, records: list[dict]) -> list[dict]:
    name = INDICATORS[indicator_code]
    rows = []
    for record in records:
        iso = record.get("countryiso3code") or ""
        year_raw = record.get("date")
        if not iso or year_raw is None:
            continue
        rows.append(
            {
                "country": record.get("country", {}).get("value"),
                "iso": iso,
                "year": int(year_raw),
                "indicator": name,
                "indicator_code": indicator_code,
                "value": record.get("value"),
            }
        )
    if not rows:
        raise RuntimeError(f"Empty series after parsing indicator {indicator_code}")
    return rows


def fetch_panel() -> pd.DataFrame:
    meta = fetch_country_meta()
    rows: list[dict] = []
    for code in INDICATORS:
        records = fetch_indicator(code, "all")
        records.extend(fetch_indicator(code, WORLD_ISO))
        rows.extend(rows_from_payload(code, records))

    frame = pd.DataFrame(rows)
    frame = frame.merge(meta[["iso", "region", "income_level", "is_aggregate"]], on="iso", how="left")
    frame["is_aggregate"] = frame["is_aggregate"].fillna(frame["iso"] == WORLD_ISO)
    frame["is_focus"] = frame["iso"].isin(FOCUS_ISOS)
    frame["is_world"] = frame["iso"] == WORLD_ISO
    frame = frame.drop_duplicates(subset=["iso", "year", "indicator"])
    frame = frame.sort_values(["iso", "year", "indicator"]).reset_index(drop=True)

    countries = frame.loc[~frame["is_aggregate"], "iso"].nunique()
    if countries < MIN_COUNTRIES:
        raise RuntimeError(f"Expected many countries, found {countries}")

    missing = frame.groupby("indicator")["value"].apply(lambda s: s.notna().sum())
    empty = missing[missing == 0]
    if not empty.empty:
        raise RuntimeError(f"No non-null values for: {', '.join(empty.index)}")

    return frame


def save_panel(frame: pd.DataFrame, path: Path = CACHE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def cache_looks_complete(frame: pd.DataFrame) -> bool:
    return frame["iso"].nunique() >= MIN_COUNTRIES


def load_or_fetch(path: Path = CACHE_PATH, refresh: bool = False) -> pd.DataFrame:
    if path.exists() and not refresh:
        frame = pd.read_parquet(path)
        if cache_looks_complete(frame):
            return frame
    frame = fetch_panel()
    save_panel(frame, path)
    return frame


def main() -> None:
    frame = load_or_fetch(refresh=True)
    countries = frame.loc[~frame["is_aggregate"], "iso"].nunique()
    print(f"Wrote {CACHE_PATH} ({len(frame)} rows, {countries} countries)")
    print(frame.groupby("indicator")["value"].agg(["count", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
