"""Thin FastAPI surface over the same facts packet and briefing functions."""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from brief import answer_question, generate_briefing
from compute import build_facts
from fetch_world_bank import load_panel
from schemas import AskResponse, Briefing, FactsPacket

app = FastAPI(
    title="Macro briefing",
    description="Facts packet and grounded LLM briefing over World Bank indicators.",
    version="0.1.0",
)


def get_packet() -> FactsPacket:
    use_fixture = os.getenv("USE_FIXTURE") == "1"
    return build_facts(load_panel(use_fixture=use_fixture))


class AskBody(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/facts")
def facts(packet: FactsPacket = Depends(get_packet)) -> FactsPacket:
    return packet


@app.get("/briefing")
def briefing(
    skip_llm: bool = False,
    packet: FactsPacket = Depends(get_packet),
) -> Briefing:
    return generate_briefing(packet, skip=skip_llm, cache=False)


@app.post("/ask")
def ask(body: AskBody, packet: FactsPacket = Depends(get_packet)) -> AskResponse:
    return answer_question(packet, body.question)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
