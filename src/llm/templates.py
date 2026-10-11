"""Deterministic template explanation. Owner: P4.

Used when the LLM is unavailable or its output fails validation twice, and when running with no
API key and no cache hit (D10). It is also the baseline the LLM is compared against in the eval.
Returns the same {"headline", "text"} dict as the LLM path (P5 contract), uses only the payload
built from the row, and is tested to always pass the validator.
"""
from __future__ import annotations

from collections.abc import Mapping

HEADLINES = {
    "EV_FINANCE": "An EV could cut your transport emissions",
    "GREEN_HOME": "Energy upgrades could make your home greener",
    "SOLAR_FINANCE": "Solar panels could power your home more cleanly",
    "PAPERLESS_BANKING": "Going paperless could cut paper waste",
}
DEFAULT_HEADLINE = "A greener option that may suit you"
MAX_EVIDENCE_ITEMS = 2


def join_phrases(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def template_explanation(row: Mapping, payload: Mapping | None = None) -> dict:
    from .payload import build_payload  # local import keeps this module dependency-light
    from .validator import MAX_TEXT_WORDS, word_count

    payload = payload if payload is not None else build_payload(row)
    reasons = [r["meaning"] for r in payload["reasons"]]
    impact = payload["estimated_impact"]

    def compose(n_evidence: int) -> str:
        sentences = []
        if payload["certainty"] == "low":
            sentences.append("Based on the limited data we have, this may be relevant to you.")
        if reasons:
            sentences.append(f"We noticed {join_phrases(reasons)}.")
        evidence = payload["evidence"][:n_evidence]
        if evidence:
            sentences.append("Key figures: " + "; ".join(f"{e['label']} {e['value']}" for e in evidence) + ".")
        if impact["status"] == "ESTIMATED":
            sentences.append(f"Our estimate is that {payload['topic']} could avoid {impact['text']}.")
        else:
            sentences.append("We can't estimate the environmental impact for you yet.")
        return " ".join(sentences)

    # Every reason must stay covered, so long cards drop key figures first.
    for n_evidence in range(MAX_EVIDENCE_ITEMS, -1, -1):
        text = compose(n_evidence)
        if word_count(text) <= MAX_TEXT_WORDS:
            break
    return {"headline": HEADLINES.get(row.get("opportunity"), DEFAULT_HEADLINE), "text": text}


def business_template(payload: Mapping) -> dict:
    return {"summary": f"Key metrics for {payload['segment']}.",
            "bullets": [f"{m['label'][0].upper()}{m['label'][1:]}: {m['value']}" for m in payload["metrics"][:4]]}
