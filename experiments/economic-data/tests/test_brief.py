from __future__ import annotations

from brief import ground_highlights, skipped_briefing
from compute import build_facts
from schemas import Briefing, BriefingHighlight, CountryNote


def test_grounding_keeps_matching_highlights(panel):
    packet = build_facts(panel)
    usa = next(row for row in packet.focus if row.iso == "USA")
    inflation = usa.metrics["inflation"].value
    assert inflation is not None
    briefing = Briefing(
        executive_summary="Test",
        country_notes=[CountryNote(iso="USA", headline="ok", detail="ok")],
        highlights=[
            BriefingHighlight(
                iso="USA",
                metric="inflation",
                claimed_value=inflation,
                why_it_matters="copied from packet",
            ),
            BriefingHighlight(
                iso="USA",
                metric="inflation",
                claimed_value=inflation + 9,
                why_it_matters="invented",
            ),
            BriefingHighlight(
                iso="ZZZ",
                metric="inflation",
                claimed_value=1.0,
                why_it_matters="unknown iso",
            ),
        ],
    )
    grounded = ground_highlights(packet, briefing)
    assert len(grounded.highlights) == 1
    assert grounded.highlights[0].claimed_value == inflation
    assert grounded.dropped_highlights == 2
    assert grounded.grounded is False


def test_skipped_briefing_reason():
    briefing = skipped_briefing("OPENAI_API_KEY is not set", "gpt-4o-mini")
    assert briefing.skipped_reason
    assert briefing.executive_summary == ""
    assert briefing.highlights == []
