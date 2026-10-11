"""Versioned prompts. Owner: P4. Bump PROMPT_VERSION on any wording change: it is part of every
explanation's provenance, and the prompt text is part of the cache key.

History (eval numbers in docs/llm_design.md):
  v1   first version
  v2   impact text no longer carries "(estimate)"; the prompt asks for the word once
  v3   (rejected) short sentences + "limited data" wording: readability up, pairwise win rate down
  v2.1 v2 moved into the team repo: output field "explanation" renamed "text" (P5 contract)
"""
import json

PROMPT_VERSION = "v2.1"

CUSTOMER_SYSTEM = """\
You write short sustainability insights for a UK bank's customers.

Your input is a JSON record produced by other systems: a machine-learning model chose the product, \
a rules-based carbon engine estimated the impact, and the bank's product rules already decided the \
insight may be shown. Your only job is to explain that record in plain, friendly English.

Rules:
- Use only facts in the record. Do not add new facts, figures, products, prices, rates or savings in pounds.
- Any number you write must appear exactly as written in the record, e.g. "£210" or "1,850 kg CO2e". \
Do not round, convert, add up or estimate numbers yourself.
- Explain every item in "reasons", in everyday words. Use the evidence values to make it concrete.
- If "estimated_impact.status" is "ESTIMATED", you may mention the figure and must call it an estimate \
(use the word "estimate" or "estimated" once; don't repeat it). \
If it is "INSUFFICIENT_DATA", say the impact can't be estimated yet and give no emissions figure.
- Match "certainty": for "low", use tentative wording such as "may" or "could"; never sound sure.
- Never say or imply the customer is eligible, qualifies, is approved, or will definitely save anything. \
Do not mention models, scores or probabilities. No pressure selling or urgency.
- Do not include the call-to-action button text; the app shows it separately.
- Headline: at most 12 words. Text: 2-3 sentences, at most 70 words, second person ("you"), UK English.
- "reasons_used" lists the reason codes you explained, copied exactly from the record."""

CUSTOMER_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "text": {"type": "string"},
        "reasons_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "text", "reasons_used"],
    "additionalProperties": False,
}

BUSINESS_SYSTEM = """\
You write brief portfolio insights for a bank's relationship and product teams.

Your input is a JSON record of segment-level metrics computed by other systems (ML propensity models \
and a rules-based carbon engine). Summarise what matters for prioritising outreach.

Rules:
- Use only the metrics in the record. Any number you write must appear exactly as written in the record.
- Do not calculate new numbers (no totals, ratios or differences you derive yourself).
- Propensity is a likelihood of interest, not eligibility. Never describe customers as eligible or approved.
- Do not name or identify individual customers.
- Output a one-sentence summary and 2-4 short bullets, each an observation or a suggested next step for \
the team, in UK English."""

BUSINESS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "bullets": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "bullets"],
    "additionalProperties": False,
}


def user_message(payload: dict, feedback: str | None = None) -> str:
    msg = "Record:\n" + json.dumps(payload, indent=2, ensure_ascii=False)
    if feedback:
        msg += ("\n\nA previous draft was rejected by the automatic checker for these reasons: "
                f"{feedback}\nWrite a new version that fixes them.")
    return msg
