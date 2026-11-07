"""Deterministic facts packet: snapshots, OLS, and a sklearn inflation baseline."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from panels import INDICATORS, latest_year_with_coverage, to_wide
from schemas import (
    CountryFacts,
    CountryForecast,
    FactsPacket,
    ForecastFacts,
    MetricValue,
    RegressionFacts,
    RegressionTerm,
)

FLAG_INFLATION_HIGH = "inflation_elevated"
FLAG_GROWTH_NEG = "growth_contraction"
FLAG_UNEMP_HIGH = "unemployment_elevated"
Z_FLAG = 1.5
HOLD_OUT_YEARS = 4
FEATURE_COLS = ["inflation_lag1", "unemployment_lag1", "gdp_growth_lag1"]


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _zscore(history: pd.Series, value: float) -> float | None:
    series = history.dropna()
    if len(series) < 5:
        return None
    std = float(series.std(ddof=1))
    if std == 0 or pd.isna(std):
        return None
    return (value - float(series.mean())) / std


def _country_facts(sub: pd.DataFrame) -> CountryFacts:
    sub = sub.sort_values("year")
    latest_year = int(sub.dropna(subset=INDICATORS, how="all")["year"].max())
    metrics: dict[str, MetricValue] = {}
    flags: list[str] = []

    for col in INDICATORS:
        observed = sub.dropna(subset=[col])
        if observed.empty:
            metrics[col] = MetricValue()
            continue
        last = observed.iloc[-1]
        year = int(last["year"])
        value = float(last[col])
        prev = observed[observed["year"] == year - 1]
        yoy = None if prev.empty else float(value - float(prev.iloc[0][col]))
        zscore = _zscore(sub[col], value)
        metrics[col] = MetricValue(
            value=_round(value),
            year=year,
            yoy_change=_round(yoy),
            zscore=_round(zscore),
        )
        if col == "inflation" and zscore is not None and zscore >= Z_FLAG:
            flags.append(FLAG_INFLATION_HIGH)
        if col == "gdp_growth" and value < 0:
            flags.append(FLAG_GROWTH_NEG)
        if col == "unemployment" and zscore is not None and zscore >= Z_FLAG:
            flags.append(FLAG_UNEMP_HIGH)

    row = sub.iloc[-1]
    return CountryFacts(
        iso=str(row["iso"]),
        country=str(row["country"]),
        latest_year=latest_year,
        metrics=metrics,
        flags=flags,
    )


def _regression(focus: pd.DataFrame) -> RegressionFacts:
    model_df = focus.dropna(subset=["inflation", "unemployment"])
    model = smf.ols("inflation ~ unemployment + C(country)", data=model_df).fit()
    terms = [
        RegressionTerm(
            term=str(name),
            coefficient=_round(float(model.params[name]), 3) or 0.0,
            std_error=_round(float(model.bse[name]), 3) or 0.0,
            p_value=_round(float(model.pvalues[name]), 3) or 0.0,
        )
        for name in model.params.index
    ]
    return RegressionFacts(
        formula="inflation ~ unemployment + C(country)",
        n=int(model.nobs),
        r_squared=_round(float(model.rsquared), 3) or 0.0,
        terms=terms,
    )


def _lagged_panel(focus: pd.DataFrame) -> pd.DataFrame:
    frame = focus.sort_values(["iso", "year"]).copy()
    for col in INDICATORS:
        frame[f"{col}_lag1"] = frame.groupby("iso")[col].shift(1)
    return frame


def _forecast(focus: pd.DataFrame) -> ForecastFacts:
    frame = _lagged_panel(focus)
    model_df = frame.dropna(subset=["inflation", *FEATURE_COLS])
    if model_df.empty:
        raise RuntimeError("Not enough lagged rows to fit the inflation model")

    max_year = int(model_df["year"].max())
    cutoff = max_year - HOLD_OUT_YEARS
    train = model_df[model_df["year"] <= cutoff]
    test = model_df[model_df["year"] > cutoff]
    if train.empty or test.empty:
        cutoff = int(model_df["year"].quantile(0.7))
        train = model_df[model_df["year"] <= cutoff]
        test = model_df[model_df["year"] > cutoff]
    if train.empty or test.empty:
        raise RuntimeError("Could not form a time-aware train/test split")

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(train[FEATURE_COLS], train["inflation"])
    predicted = pipe.predict(test[FEATURE_COLS])
    test_mae = float(mean_absolute_error(test["inflation"], predicted))
    naive_mae = float(mean_absolute_error(test["inflation"], test["inflation_lag1"]))

    predictions: list[CountryForecast] = []
    horizon_year = int(focus["year"].max()) + 1
    for iso, sub in frame.groupby("iso"):
        usable = sub.dropna(subset=FEATURE_COLS)
        if usable.empty:
            continue
        last = usable.iloc[-1]
        if pd.isna(last["inflation"]):
            continue
        pred = float(pipe.predict(last[FEATURE_COLS].to_frame().T)[0])
        predictions.append(
            CountryForecast(
                iso=str(iso),
                country=str(last["country"]),
                last_year=int(last["year"]),
                last_inflation=_round(float(last["inflation"])) or 0.0,
                predicted_inflation=_round(pred) or 0.0,
                naive_inflation=_round(float(last["inflation"])) or 0.0,
            )
        )
    predictions.sort(key=lambda row: row.country)

    train_lo, train_hi = int(train["year"].min()), int(train["year"].max())
    test_lo, test_hi = int(test["year"].min()), int(test["year"].max())
    return ForecastFacts(
        target="inflation",
        horizon_year=horizon_year,
        model="Ridge (standardized lags)",
        features=list(FEATURE_COLS),
        train_years=f"{train_lo}–{train_hi}",
        test_years=f"{test_lo}–{test_hi}",
        test_mae=_round(test_mae, 3) or 0.0,
        naive_mae=_round(naive_mae, 3) or 0.0,
        beats_naive=test_mae < naive_mae,
        n_train=int(len(train)),
        n_test=int(len(test)),
        predictions=predictions,
    )


def _correlations(focus: pd.DataFrame) -> dict[str, dict[str, float]]:
    corr = focus[INDICATORS].corr()
    out: dict[str, dict[str, float]] = {}
    for row in INDICATORS:
        out[row] = {
            col: float(round(corr.loc[row, col], 3)) for col in INDICATORS
        }
    return out


def build_facts(long: pd.DataFrame) -> FactsPacket:
    wide = to_wide(long)
    focus = wide[wide["is_focus"]].copy()
    countries = wide[~wide["is_aggregate"]].copy()
    world = wide[wide["is_world"]].copy()
    if world.empty:
        raise RuntimeError("Facts packet requires the World (WLD) series")
    if focus["iso"].nunique() < 6:
        raise RuntimeError("Facts packet requires the six-country comparison set")

    snapshot_year = latest_year_with_coverage(countries, "inflation")
    return FactsPacket(
        year_min=int(wide["year"].min()),
        year_max=int(wide["year"].max()),
        n_countries=int(countries["iso"].nunique()),
        snapshot_year=int(snapshot_year),
        world=_country_facts(world),
        focus=[
            _country_facts(focus[focus["iso"] == iso])
            for iso in sorted(focus["iso"].unique())
        ],
        correlations=_correlations(focus),
        regression=_regression(focus),
        forecast=_forecast(focus),
    )


def country_lookup(packet: FactsPacket) -> dict[str, CountryFacts]:
    mapping = {row.iso: row for row in packet.focus}
    mapping[packet.world.iso] = packet.world
    return mapping


def metric_value(packet: FactsPacket, iso: str, metric: str) -> float | None:
    country = country_lookup(packet).get(iso)
    if country is None:
        return None
    item = country.metrics.get(metric)
    return None if item is None else item.value


def iter_numeric_tokens(packet: FactsPacket) -> Iterable[float]:
    yield from (
        packet.year_min,
        packet.year_max,
        packet.n_countries,
        packet.snapshot_year,
        packet.regression.n,
        packet.regression.r_squared,
        packet.forecast.test_mae,
        packet.forecast.naive_mae,
        packet.forecast.horizon_year,
    )
    for term in packet.regression.terms:
        yield term.coefficient
        yield term.p_value
    for country in (packet.world, *packet.focus):
        for metric in country.metrics.values():
            if metric.value is not None:
                yield metric.value
            if metric.yoy_change is not None:
                yield metric.yoy_change
    for pred in packet.forecast.predictions:
        yield pred.last_inflation
        yield pred.predicted_inflation
