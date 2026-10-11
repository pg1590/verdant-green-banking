"""FR5 LLM layer. Owner: P4.

Plugs into P5's service as the insight provider (docs/integration.md):

    RecommendationService(bundle, insight_provider=explain)

`explain` receives ONE final eligible recommendation row (built by src/recommend/engine.py) and
returns {"headline": str, "text": str}, both nonempty (agreed with P5). The action button comes from
the catalogue's `action` field, not from this text. The LLM never sees raw transactions, IDs or scores.

Flow: payload (minimised) -> cache lookup -> LLM -> validator.validate()
      -> on failure retry once with the checker's feedback -> on failure templates.template_explanation()
Cache: settings.llm.cache_dir, committed so judges can rerun without a key (D10).

    python -m src.llm.explain [bundle.json]     # pre-generate explanations for every eligible row
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from src.common import config

from . import prompts
from .client import LLMUnavailable, get_provider, llm_settings
from .payload import build_payload, format_evidence, withheld_evidence_keys
from .templates import business_template, template_explanation
from .validator import feedback, validate, validate_business

log = logging.getLogger(__name__)
PROMPT_VERSION = prompts.PROMPT_VERSION
DEFAULT_BUNDLE = config.PROCESSED_DIR / "team_bundle.json"
OUTPUT_PATH = config.OUTPUTS_DIR / "llm_explanations.jsonl"

_provider = None


def _default_provider():
    global _provider
    if _provider is None:
        _provider = get_provider()
    return _provider


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def explain_with_provenance(row: Mapping, provider=None) -> dict | None:
    """Explanation plus an audit trail. Returns None for a suppressed row (P5 never sends one)."""
    if not row.get("eligible_to_show"):
        return None
    provider = provider or _default_provider()
    payload = build_payload(row)
    attempts, output, result, source, error, fb = [], None, None, "TEMPLATE", None, None

    for attempt in range(1, 2 + int(llm_settings().get("max_retries", 1))):
        try:
            candidate = provider.complete_json(prompts.CUSTOMER_SYSTEM, prompts.user_message(payload, fb),
                                               prompts.CUSTOMER_SCHEMA)
        except LLMUnavailable as e:
            error = str(e)
            break
        check = validate(candidate, row, payload)
        attempts.append({"attempt": attempt, "ok": check["ok"], "failures": check["failures"],
                         "meta": getattr(provider, "last_meta", {})})
        if check["ok"]:
            output, result, source = candidate, check, "LLM"
            break
        fb = feedback(check)

    if output is None:
        output = template_explanation(row, payload)
        result = validate(output, row, payload)
        if not result["ok"]:  # never expected; covered by tests
            log.error("template failed validation for %s/%s: %s",
                      row.get("customer_id"), row.get("product_code"), feedback(result))

    return {
        "headline": output["headline"].strip(),
        "text": output["text"].strip(),
        "provenance": {
            "text_source": source,
            "provider": provider.name,
            "model": getattr(provider, "model", None),
            "prompt_version": PROMPT_VERSION,
            "attempts": attempts,
            "llm_error": error,
            "withheld_evidence": withheld_evidence_keys(row),
            "validation": result,
            "generated_at": _now(),
        },
    }


def explain(row: Mapping) -> dict | None:
    """P5's insight_provider: {"headline", "text"} for one eligible row (None if suppressed).

    Never raises: P5 lets provider exceptions propagate, so any unexpected error falls back to the
    validated template, and if even that fails, returns None so P5 shows its own flagged fallback
    (INVALID_PROVIDER_OUTPUT) instead of crashing the dashboard.
    """
    try:
        card = explain_with_provenance(row)
        return None if card is None else {"headline": card["headline"], "text": card["text"]}
    except Exception:
        log.exception("explain failed for %s/%s; trying the template", row.get("customer_id"), row.get("product_code"))
    try:
        fallback = template_explanation(row)
        if validate(fallback, row)["ok"]:
            return fallback
    except Exception:
        log.exception("template failed for %s/%s", row.get("customer_id"), row.get("product_code"))
    return None


def build_business_payload(segment: Mapping) -> dict:
    """From one row of P5's RecommendationService.segments(); customer IDs are dropped."""
    product = segment.get("product_name") or segment.get("product_code", "")
    metrics = {"customers_count": segment.get("customers"),
               "mean_propensity_pct": None if segment.get("mean_propensity") is None else round(100 * segment["mean_propensity"]),
               "mean_confidence_pct": None if segment.get("mean_confidence") is None else round(100 * segment["mean_confidence"])}
    return {"segment": f"{segment.get('region', 'All regions')}, {product}",
            "metrics": [format_evidence(k, v) for k, v in metrics.items() if v is not None]}


def business_insight(segment: Mapping, provider=None) -> dict:
    """FR6 RM-facing summary of one segment: {"summary", "bullets", "provenance"}."""
    provider = provider or _default_provider()
    payload = build_business_payload(segment)
    output, source, fb, error = None, "TEMPLATE", None, None
    for _ in range(1 + int(llm_settings().get("max_retries", 1))):
        try:
            candidate = provider.complete_json(prompts.BUSINESS_SYSTEM, prompts.user_message(payload, fb),
                                               prompts.BUSINESS_SCHEMA)
        except LLMUnavailable as e:
            error = str(e)
            break
        check = validate_business(candidate, payload)
        if check["ok"]:
            output, source = candidate, "LLM"
            break
        fb = feedback(check)
    output = output or business_template(payload)
    return {**output, "segment": payload["segment"],
            "provenance": {"text_source": source, "model": getattr(provider, "model", None),
                           "prompt_version": PROMPT_VERSION, "llm_error": error, "generated_at": _now()}}


def main(bundle_path: str | Path | None = None) -> None:
    """Pipeline stage: explain every eligible row of the team bundle; warms the committed cache."""
    from src.recommend.service import RecommendationService

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = Path(bundle_path or DEFAULT_BUNDLE)
    if not path.exists():
        print(f"    no team bundle at {path}; nothing to explain yet (P5 exports it after P1-P3 outputs land)")
        return
    service = RecommendationService.from_file(path)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for cid in service.customers:
            for row in service.recommendations(cid)["recommendations"]:
                card = explain_with_provenance(row)
                counts[card["provenance"]["text_source"]] += 1
                f.write(json.dumps({"customer_id": cid, "product_code": row["product_code"], **card},
                                   ensure_ascii=False) + "\n")
    print(f"    wrote {OUTPUT_PATH.relative_to(config.ROOT)}: {dict(counts)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
