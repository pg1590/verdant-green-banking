"""Post-generation checks on LLM output. Owner: P4.

The explanation {"headline", "text"} is shown only if every hard check passes:
  schema               headline and text are nonempty strings (P5 contract)
  ungrounded_numbers   every number appears in the payload ("CO2" digits ignored; no rounding allowed)
  banned_phrases       eligibility/approval, guarantees, model internals, pressure selling,
                       directive advice, customer IDs (eligible to show never means approved for credit)
  invented_reason_codes  reasons_used must be a subset of the row's reason codes
  reason_coverage      each reason code is mentioned (keywords in src/llm/phrases.py)
  impact_label         an impact figure used in the text must be called an estimate
  insufficient_impact  no emissions figure or claim when impact is INSUFFICIENT_DATA
  length               headline <= 12 words, text <= 80 words
Soft check (recorded only): low_certainty_hedge.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from . import phrases
from .payload import build_payload

MAX_HEADLINE_WORDS = 12
MAX_TEXT_WORDS = 80

# (name, pattern), matched case-insensitively.
BANNED_PATTERNS = [
    ("eligibility_claim", r"\beligib\w*|\bqualif(y|ies|ied)\b|\b(pre-?)?approv(ed|al)\b"),
    ("guarantee", r"\bguarantee\w*|\bdefinitely\b|\bcertainly will\b|\bwill save\b|\brisk-free\b"),
    ("model_internals", r"\bpropensity\b|\bprobability\b|\bconfidence score\b|\bour (model|algorithm)\b|\bmachine learning\b"),
    ("pressure_selling", r"\bact now\b|\blimited time\b|\bdon'?t miss\b|\bhurry\b|\bbefore it'?s too late\b"),
    ("directive_advice", r"\byou (must|should definitely|need to) (take out|apply|borrow)\b"),
]
HEDGE_WORDS = re.compile(r"\b(may|might|could|possibly|perhaps|worth considering|if)\b", re.I)
CUSTOMER_ID = re.compile(r"\bC\d{4,}\b")
_FORMULA = re.compile(r"CO[2₂]e?", re.I)  # so "CO2e" doesn't count as the number 2
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def extract_numbers(text: str) -> set[float]:
    out = set()
    for tok in _NUMBER.findall(_FORMULA.sub(" ", text)):
        try:
            out.add(float(tok.rstrip(",.").replace(",", "")))
        except ValueError:
            continue
    return out


def _strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, Mapping):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def allowed_numbers(payload: Mapping) -> set[float]:
    return set().union(*(extract_numbers(s) for s in _strings(payload)))


def mentions_reason(lower_text: str, code: str) -> bool:
    """Keywords match at a word start, so 'renovat' matches 'renovation' but 'end' doesn't match 'spend'."""
    return any(re.search(r"\b" + re.escape(k), lower_text) for k in phrases.keywords(code))


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text))


def _syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    return max(len(groups) - (1 if word.lower().endswith("e") and len(groups) > 1 else 0), 1)


def flesch_reading_ease(text: str) -> float:
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return 0.0
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    syl = sum(_syllables(w) for w in words)
    return round(206.835 - 1.015 * len(words) / sentences - 84.6 * syl / len(words), 1)


def validate(output: Mapping, row: Mapping, payload: Mapping | None = None) -> dict:
    """Return {"ok": bool, "failures": [{"check", "detail"}], "warnings": [...], "metrics": {...}}."""
    payload = payload if payload is not None else build_payload(row)
    failures, warnings = [], []

    def fail(check, detail):
        failures.append({"check": check, "detail": detail})

    headline = output.get("headline") if isinstance(output, Mapping) else None
    text = output.get("text") if isinstance(output, Mapping) else None
    if not isinstance(headline, str) or not headline.strip() or not isinstance(text, str) or not text.strip():
        fail("schema", "headline and text must be nonempty strings")
        return {"ok": False, "failures": failures, "warnings": warnings, "metrics": {}}
    full = f"{headline}\n{text}"
    lower = full.lower()

    ungrounded = sorted(n for n in extract_numbers(full) if n not in allowed_numbers(payload))
    if ungrounded:
        fail("ungrounded_numbers", f"numbers not in input: {ungrounded}")

    for name, pattern in BANNED_PATTERNS:
        m = re.search(pattern, full, re.I)
        if m:
            fail("banned_phrases", f"{name}: '{m.group(0)}'")
    if CUSTOMER_ID.search(full):
        fail("banned_phrases", "contains a customer identifier")

    codes = list(row.get("reason_codes") or [])
    invented = sorted(set(output.get("reasons_used") or []) - set(codes))
    if invented:
        fail("invented_reason_codes", f"codes not in input: {invented}")
    uncovered = [c for c in codes if not mentions_reason(lower, c)]
    if uncovered:
        fail("reason_coverage", f"text does not mention: {uncovered}")

    impact = payload["estimated_impact"]
    if impact["status"] == "ESTIMATED":
        if extract_numbers(full) & extract_numbers(impact["text"]) and "estimat" not in lower:
            fail("impact_label", "impact figure used without calling it an estimate")
    elif re.search(r"\bkg\b|\btonnes?\b|\bco[2₂]e?\b.*\b(cut|reduc|sav)", lower):
        fail("insufficient_impact", "states an emissions figure although impact data is insufficient")

    if word_count(headline) > MAX_HEADLINE_WORDS:
        fail("length", f"headline over {MAX_HEADLINE_WORDS} words")
    if word_count(text) > MAX_TEXT_WORDS:
        fail("length", f"text over {MAX_TEXT_WORDS} words")

    if payload["certainty"] == "low" and not HEDGE_WORDS.search(full):
        warnings.append({"check": "low_certainty_hedge", "detail": "no tentative wording for a low-certainty insight"})

    metrics = {
        "numbers_total": len(extract_numbers(full)),
        "numbers_ungrounded": len(ungrounded),
        "reason_coverage": round(1 - len(uncovered) / (len(codes) or 1), 3),
        "text_words": word_count(text),
        "flesch_reading_ease": flesch_reading_ease(text),
    }
    return {"ok": not failures, "failures": failures, "warnings": warnings, "metrics": metrics}


def feedback(result: Mapping) -> str:
    return "; ".join(f"{f['check']}: {f['detail']}" for f in result["failures"])


def validate_business(output: Mapping, payload: Mapping) -> dict:
    bullets = output.get("bullets") if isinstance(output, Mapping) else None
    if not isinstance(bullets, list) or not bullets or not all(isinstance(b, str) and b.strip() for b in bullets):
        return {"ok": False, "failures": [{"check": "schema", "detail": "bullets must be a nonempty list of strings"}]}
    text = "\n".join([output.get("summary", "")] + bullets)
    failures = []
    ungrounded = sorted(n for n in extract_numbers(text) if n not in allowed_numbers(payload))
    if ungrounded:
        failures.append({"check": "ungrounded_numbers", "detail": f"numbers not in input: {ungrounded}"})
    for name, pattern in BANNED_PATTERNS[:2]:  # eligibility and guarantee claims are banned for RMs too
        m = re.search(pattern, text, re.I)
        if m:
            failures.append({"check": "banned_phrases", "detail": f"{name}: '{m.group(0)}'"})
    if CUSTOMER_ID.search(text):
        failures.append({"check": "banned_phrases", "detail": "contains a customer identifier"})
    return {"ok": not failures, "failures": failures}
