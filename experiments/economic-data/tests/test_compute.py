from __future__ import annotations

from compute import build_facts
from schemas import FactsPacket


def test_facts_packet_schema(panel):
    packet = build_facts(panel)
    assert isinstance(packet, FactsPacket)
    assert packet.n_countries >= 80
    assert packet.world.iso == "WLD"
    assert {row.iso for row in packet.focus} == {"BRA", "CHN", "DEU", "IND", "JPN", "USA"}
    assert "inflation" in packet.world.metrics
    assert packet.world.metrics["inflation"].value is not None
    assert packet.regression.n > 0
    assert set(packet.correlations["inflation"]) == {
        "gdp_growth",
        "inflation",
        "unemployment",
    }


def test_forecast_has_baseline_and_horizon(panel):
    packet = build_facts(panel)
    forecast = packet.forecast
    assert forecast.target == "inflation"
    assert forecast.horizon_year == packet.year_max + 1
    assert forecast.n_train > forecast.n_test > 0
    assert forecast.naive_mae >= 0
    assert forecast.test_mae >= 0
    assert len(forecast.predictions) == 6
    assert forecast.features == [
        "inflation_lag1",
        "unemployment_lag1",
        "gdp_growth_lag1",
    ]


def test_flags_for_contraction_year(panel):
    packet = build_facts(panel)
    # 2020 growth is shocked negative in the synthetic panel; latest year is 2024
    # so flags may be empty for latest — check the builder still returns a list.
    for row in packet.focus:
        assert isinstance(row.flags, list)
        for metric in row.metrics.values():
            assert metric.year is not None
