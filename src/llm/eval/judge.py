"""LLM-as-judge for customer explanations. Owner: P4.

Uses a different model family from the generator (Gemma 4 vs Gemini) so the generator isn't grading
itself; settings.llm.judge_provider / judge_models. Two methods:
  score     1-5 rubric on the card exactly as the customer sees it
  pairwise  LLM card vs template card for the same row, order randomised to cancel position bias
            (1-5 scores hit the ceiling for both; pairwise discriminates)
LLM_JUDGE_PROVIDER=groq (or run_eval --judge-provider groq) uses an open-weight family on Groq
(settings llm.groq_judge_models, default openai/gpt-oss-120b) as a second, independent judge;
run_eval then writes separate *_judge-groq reports.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping

from ..client import AnthropicProvider, GeminiProvider, GroqProvider, llm_settings
from ..payload import impact_display

JUDGE_PROVIDER = os.getenv("LLM_JUDGE_PROVIDER") or llm_settings().get("judge_provider") or "gemini"
JUDGE_MODELS = (os.getenv("LLM_JUDGE_MODELS", "").split(",") if os.getenv("LLM_JUDGE_MODELS")
                else llm_settings().get("groq_judge_models") or ["openai/gpt-oss-120b"] if JUDGE_PROVIDER == "groq"
                else llm_settings().get("judge_models") or ["gemma-4-26b-a4b-it"])
JUDGE_MODEL = JUDGE_MODELS[0]
CRITERIA = ["faithfulness", "clarity", "personalisation", "tone", "actionability"]

JUDGE_SYSTEM = """\
You grade short customer-facing sustainability insights written by a bank's assistant.
You get the source RECORD (the only facts the writer was allowed to use) and the INSIGHT CARD exactly as the
customer sees it. The impact line, data date and action button on the card are added by the app from the RECORD.

Score each criterion from 1 (poor) to 5 (excellent):
- faithfulness: every claim is supported by the RECORD; nothing invented or exaggerated.
- clarity: plain English a non-expert understands on first read.
- personalisation: uses this customer's reasons and evidence rather than generic text.
- tone: friendly and neutral; no pressure, no implied approval or guaranteed savings.
- actionability: makes clear why this matters and what the customer could consider next.

List each unsupported claim verbatim in "unsupported_claims" (empty list if none). Be strict."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {**{c: {"type": "integer"} for c in CRITERIA},
                   "unsupported_claims": {"type": "array", "items": {"type": "string"}}},
    "required": CRITERIA + ["unsupported_claims"],
    "additionalProperties": False,
}

PAIRWISE_SYSTEM = """\
You compare two versions (A and B) of the same customer-facing sustainability insight from a bank.
Both were written from the same RECORD. Decide which one a typical customer would find clearer, more
relevant to them personally, and more useful, while staying faithful to the RECORD.
A version with any claim not supported by the RECORD loses. Answer "tie" only if they are genuinely equal."""

PAIRWISE_SCHEMA = {
    "type": "object",
    "properties": {"winner": {"type": "string", "enum": ["A", "B", "tie"]}, "reason": {"type": "string"}},
    "required": ["winner", "reason"],
    "additionalProperties": False,
}


def card_as_seen(card: Mapping, row: Mapping) -> str:
    """The explanation plus what P5's UI renders from the row around it."""
    impact = impact_display(row.get("impact"))
    impact_line = (f"Potential annual impact: {impact['text']} (estimate)" if impact["status"] == "ESTIMATED"
                   else "Potential annual impact: not enough data to estimate yet")
    action = (row.get("action") or {}).get("label", "")
    return (f"Headline: {card['headline']}\nText: {card['text']}\n{impact_line}\n"
            f"Data as of {row.get('data_timestamp')}\n[Button] {action}")


class Judge:
    def __init__(self, provider: str | None = None):
        provider = provider or JUDGE_PROVIDER
        if provider == "anthropic":
            self.provider = AnthropicProvider(model=JUDGE_MODEL, effort="medium")
        elif provider == "groq":
            self.provider = GroqProvider(models=JUDGE_MODELS if JUDGE_PROVIDER == "groq"
                                         else llm_settings().get("groq_judge_models") or ["openai/gpt-oss-120b"])
        else:
            self.provider = GeminiProvider(models=JUDGE_MODELS)
        self.name = provider
        self.model = self.provider.model

    def score(self, payload: Mapping, card: Mapping, row: Mapping) -> dict:
        user = ("RECORD:\n" + json.dumps(payload, indent=2, ensure_ascii=False)
                + "\n\nINSIGHT CARD:\n" + card_as_seen(card, row))
        out = self.provider.complete_json(JUDGE_SYSTEM, user, JUDGE_SCHEMA)
        for c in CRITERIA:
            out[c] = max(1, min(5, int(out[c])))
        return out


def pairwise(judge: Judge, payload: Mapping, card_x: Mapping, card_y: Mapping, swap: bool) -> tuple[str, str]:
    """Return ('x' | 'y' | 'tie', reason). `swap` decides which card is shown as A."""
    a, b = (card_y, card_x) if swap else (card_x, card_y)
    user = ("RECORD:\n" + json.dumps(payload, indent=2, ensure_ascii=False)
            + f"\n\nVERSION A:\nHeadline: {a['headline']}\nText: {a['text']}"
            + f"\n\nVERSION B:\nHeadline: {b['headline']}\nText: {b['text']}")
    out = judge.provider.complete_json(PAIRWISE_SYSTEM, user, PAIRWISE_SCHEMA)
    if out["winner"] == "tie":
        return "tie", out["reason"]
    a_won = out["winner"] == "A"
    return ("y" if a_won else "x") if swap else ("x" if a_won else "y"), out["reason"]
