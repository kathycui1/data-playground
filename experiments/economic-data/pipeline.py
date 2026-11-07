"""Run ingest → facts → optional LLM briefing → HTML report."""

from __future__ import annotations

import argparse
from pathlib import Path

from brief import generate_briefing
from compute import build_facts
from fetch_world_bank import load_panel
from render_report import REPORT_PATH, build_report
from schemas import Briefing, FactsPacket


def run(
    *,
    use_fixture: bool = False,
    refresh: bool = False,
    skip_llm: bool = False,
    path: Path = REPORT_PATH,
) -> tuple[Path, FactsPacket, Briefing]:
    long = load_panel(refresh=refresh, use_fixture=use_fixture)
    facts = build_facts(long)
    briefing = generate_briefing(facts, skip=skip_llm)
    report = build_report(long, facts=facts, briefing=briefing, path=path)
    return report, facts, briefing


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the economic briefing report")
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Use the committed sample parquet instead of the World Bank cache/API",
    )
    parser.add_argument("--refresh", action="store_true", help="Refetch World Bank data")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the model call and render stats only",
    )
    args = parser.parse_args()
    report, facts, briefing = run(
        use_fixture=args.fixture,
        refresh=args.refresh,
        skip_llm=args.skip_llm,
    )
    status = briefing.skipped_reason or f"briefing via {briefing.model}"
    print(report)
    print(f"{facts.n_countries} countries; {status}")


if __name__ == "__main__":
    main()
