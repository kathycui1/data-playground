from __future__ import annotations

from compute import build_facts
from fetch_world_bank import FIXTURE_PATH, load_panel


def test_committed_fixture_builds_packet():
    assert FIXTURE_PATH.exists()
    packet = build_facts(load_panel(use_fixture=True))
    assert packet.n_countries >= 80
    assert packet.world.iso == "WLD"
    assert len(packet.focus) == 6
    assert packet.forecast.predictions