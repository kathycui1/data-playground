"""Typed contracts for the facts packet and LLM briefing."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Metric = Literal["gdp_growth", "inflation", "unemployment"]


class MetricValue(BaseModel):
    value: float | None = None
    year: int | None = None
    yoy_change: float | None = None
    zscore: float | None = None


class CountryFacts(BaseModel):
    iso: str
    country: str
    latest_year: int
    metrics: dict[str, MetricValue]
    flags: list[str] = Field(default_factory=list)


class RegressionTerm(BaseModel):
    term: str
    coefficient: float
    std_error: float
    p_value: float


class RegressionFacts(BaseModel):
    formula: str
    n: int
    r_squared: float
    terms: list[RegressionTerm]


class CountryForecast(BaseModel):
    iso: str
    country: str
    last_year: int
    last_inflation: float
    predicted_inflation: float
    naive_inflation: float


class ForecastFacts(BaseModel):
    target: str = "inflation"
    horizon_year: int
    model: str
    features: list[str]
    train_years: str
    test_years: str
    test_mae: float
    naive_mae: float
    beats_naive: bool
    n_train: int
    n_test: int
    predictions: list[CountryForecast]


class FactsPacket(BaseModel):
    source: str = "World Bank Open Data"
    year_min: int
    year_max: int
    n_countries: int
    snapshot_year: int
    world: CountryFacts
    focus: list[CountryFacts]
    correlations: dict[str, dict[str, float]]
    regression: RegressionFacts
    forecast: ForecastFacts


class BriefingHighlight(BaseModel):
    model_config = ConfigDict(extra="ignore")
    iso: str
    metric: Metric
    claimed_value: float
    why_it_matters: str = ""


class CountryNote(BaseModel):
    model_config = ConfigDict(extra="ignore")
    iso: str
    headline: str
    detail: str = ""


class Briefing(BaseModel):
    model_config = ConfigDict(extra="ignore")
    executive_summary: str = ""
    country_notes: list[CountryNote] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    highlights: list[BriefingHighlight] = Field(default_factory=list)
    model: str | None = None
    grounded: bool = True
    dropped_highlights: int = 0
    skipped_reason: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    model: str | None = None
    skipped_reason: str | None = None
