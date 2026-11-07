"""Call an OpenAI-compatible chat model on the facts packet only."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

from compute import country_lookup, metric_value
from schemas import AskResponse, Briefing, BriefingHighlight, FactsPacket

load_dotenv()

PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"
BRIEFING_CACHE = PROCESSED_DIR / "briefing.json"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
VALUE_TOLERANCE = 0.15

SYSTEM_PROMPT = """You are a cautious macro analyst. You receive a JSON facts packet
computed by code. Interpret it; do not invent or round numbers differently.

Rules:
- Use only ISOs and metrics present in the packet.
- highlights[].claimed_value MUST copy packet.focus/world metrics[metric].value exactly.
- country_notes[].iso must be one of the focus ISOs.
- Prose may refer to numbers that appear in the packet, never new figures.
- No causal claims. Note lag, missingness, and that the Ridge model is a baseline.
- Be specific and short.
- At most 6 country_notes (one per focus country), 6 highlights, 4 risks, 4 follow_up_questions.

Return JSON with keys:
executive_summary (string),
country_notes (array of {iso, headline, detail}),
risks (array of strings),
follow_up_questions (array of strings),
highlights (array of {iso, metric, claimed_value, why_it_matters})
where metric is one of gdp_growth, inflation, unemployment.
Every highlight must include why_it_matters.
"""


def _settings() -> tuple[str | None, str, str]:
    key = os.getenv("OPENAI_API_KEY") or None
    base = (os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model = os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    return key, base, model


def ground_highlights(packet: FactsPacket, briefing: Briefing) -> Briefing:
    lookup = country_lookup(packet)
    kept: list[BriefingHighlight] = []
    dropped = 0
    for highlight in briefing.highlights:
        country = lookup.get(highlight.iso)
        actual = metric_value(packet, highlight.iso, highlight.metric)
        if country is None or actual is None:
            dropped += 1
            continue
        if abs(actual - highlight.claimed_value) > VALUE_TOLERANCE:
            dropped += 1
            continue
        kept.append(highlight.model_copy(update={"claimed_value": actual}))
    notes = [note for note in briefing.country_notes if note.iso in lookup]
    return briefing.model_copy(
        update={
            "highlights": kept,
            "country_notes": notes,
            "dropped_highlights": dropped,
            "grounded": dropped == 0,
        }
    )


def skipped_briefing(reason: str, model: str | None = None) -> Briefing:
    return Briefing(skipped_reason=reason, model=model, grounded=True)


def _chat(messages: list[dict], timeout: float = 60.0) -> tuple[str, str]:
    key, base, model = _settings()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()
    content = body["choices"][0]["message"]["content"]
    return content, model


def generate_briefing(
    packet: FactsPacket,
    *,
    skip: bool = False,
    cache: bool = True,
) -> Briefing:
    key, _, model = _settings()
    if skip:
        return skipped_briefing("LLM skipped by request", model)
    if not key:
        return skipped_briefing("OPENAI_API_KEY is not set", model)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Facts packet (computed; treat as source of truth):\n"
                + packet.model_dump_json()
            ),
        },
    ]
    try:
        raw, model_name = _chat(messages)
        briefing = Briefing.model_validate(json.loads(raw))
        briefing.model = model_name
        briefing.country_notes = briefing.country_notes[:6]
        briefing.highlights = briefing.highlights[:6]
        briefing.risks = briefing.risks[:4]
        briefing.follow_up_questions = briefing.follow_up_questions[:4]
        briefing = ground_highlights(packet, briefing)
    except (httpx.HTTPError, json.JSONDecodeError, ValidationError, KeyError, RuntimeError) as exc:
        return skipped_briefing(f"LLM call failed: {exc}", model)

    if cache:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        BRIEFING_CACHE.write_text(briefing.model_dump_json(indent=2), encoding="utf-8")
    return briefing


def answer_question(packet: FactsPacket, question: str) -> AskResponse:
    key, _, model = _settings()
    if not key:
        return AskResponse(
            question=question,
            answer="",
            model=model,
            skipped_reason="OPENAI_API_KEY is not set",
        )
    prompt = (
        "Answer the question using only this facts packet. "
        "If the packet does not contain the answer, say so. "
        "Return JSON {\"answer\": string}.\n\n"
        f"Question: {question}\n\n"
        f"Packet: {packet.model_dump_json()}"
    )
    try:
        raw, model_name = _chat(
            [
                {
                    "role": "system",
                    "content": "You answer from a computed economic facts packet only.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        parsed = json.loads(raw)
        answer = str(parsed.get("answer") or raw)
        return AskResponse(question=question, answer=answer, model=model_name)
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, RuntimeError) as exc:
        return AskResponse(
            question=question,
            answer="",
            model=model,
            skipped_reason=f"LLM call failed: {exc}",
        )
