# LLM layer: design, prompts and evaluation

Code: [src/llm/](../src/llm/). Tests: [tests/llm/](../tests/llm/).

## 1. Role in the system

The LLM **only explains** structured outputs. It never predicts, calculates impact or decides eligibility (brief, core rule).

```
final eligible row from P5's engine (P3 propensity + P2 impact + P5 eligibility)
  └─ minimised payload ──► cache ──► LLM (JSON-schema output) ──► validator
                                         │ fail ──► retry once with the checker's feedback ──► validator
                                         │ fail ──► deterministic template (always passes the validator)
  explain(row) -> {"headline", "text"}     (agreed with P5; the action button comes from the catalogue)
```

| What the customer sees | Produced by |
|---|---|
| Headline, 2–3 sentence text | LLM (validated) or template |
| Impact, method, data date, action button | P5's UI, from the row (never the LLM) |
| Suppressed products | Nothing. P5 never passes them, and `explain` returns `None` if it did. |

## 2. Model and API

| Item | Value (`config/settings.yaml` → `llm:`) |
|---|---|
| Provider | Google Gemini API, free tier, Python SDK `google-genai`. Key in `GEMINI_API_KEY` (`.env`, git-ignored). |
| Generator | `gemini-3.7-flash`, with fallbacks `gemini-3.6-flash → gemini-3.5-flash → gemini-3.1-flash-lite`. Free-tier models often return 503, so each gets one retry and then the next is tried. The model that actually served each response is recorded. |
| Output format | JSON-schema structured output: `headline`, `text`, `reasons_used` |
| Temperature | Provider default (`null`). Reproducibility comes from the committed cache (D10). |
| Judge | Gemma 4 (`gemma-4-26b-a4b-it`), a different model family from the generator |
| Alternative | `LLM_PROVIDER=anthropic` (`claude-opus-5-5`, effort `low`, server-side refusal fallback) |
| Cache | `data/llm_cache/` (**committed**, D10). Key = hash(provider, primary model, system prompt, payload message, schema). When a provider is selected, matching cached rows replay exactly; auto mode without a key and uncached rows use the template. |
| Failure handling | Any API error, safety block, invalid JSON or exhausted model chain leads to the template |

## 3. Data minimisation ([payload.py](../src/llm/payload.py))

The LLM sees **only**:
- the product name and topic
- reason codes with plain-English clauses ([phrases.py](../src/llm/phrases.py))
- evidence as pre-formatted display strings (`£210`, `18,400 km`, EPC `E`, flags as yes/no)
- the impact display string and method
- a certainty band (`high` ≥0.75, `medium` ≥0.5, else `low`)
- the data date

It **never** sees:
- `customer_id`, raw propensity or confidence, other products, or the action label
- free text. Any text evidence that isn't a short categorical value (≤40 characters) is withheld and logged in `provenance.withheld_evidence`. This blocks prompt injection through data fields.

## 4. Reason-code wording

P3 owns the dictionary (`config/reason_codes.yaml`: which codes exist and when they fire). P4 owns the customer wording and coverage keywords in [phrases.py](../src/llm/phrases.py).

The layer reads both dictionary formats in use: the nested `reason_codes:` map with `customer_text`, and P3's flat map with `label` and `evidence_key`. Codes P4 hasn't worded yet fall back to `customer_text`/`label`, then to the code name, so a new code never breaks the layer.

## 5. Prompts ([prompts.py](../src/llm/prompts.py))

`PROMPT_VERSION = "v2.1"`. Key instructions:
- use only the record's facts
- copy numbers exactly
- explain every reason
- call the impact an estimate once, or say it can't be estimated
- use tentative wording at low certainty
- no eligibility, approval, guarantee or pressure language
- no mention of models or scores
- no button text
- headline ≤12 words, text ≤70 words

| Version | Change | Result (dev) |
|---|---|---|
| v1 | first version | the LLM wrote "estimated … (estimate)" |
| v2 | impact display without "(estimate)"; use the word once | kept |
| v3 | short sentences, fixed order, "limited data" wording | **rejected**: readability 59.8 → 65.9, but pairwise win rate 68% → 63% and first-attempt pass 100% → 95.5% |
| v2.1 | v2 in the team repo; output field `explanation` → `text` (P5 contract) | current |

## 6. Guardrails ([validator.py](../src/llm/validator.py))

| Check | Rule |
|---|---|
| `schema` | `headline` and `text` are nonempty strings (P5 contract) |
| `ungrounded_numbers` | every number appears in the payload (CO2/CO₂ digits ignored; no rounding) |
| `banned_phrases` | eligibility/approval, guarantees ("will save"), model internals ("our model", "propensity"), pressure selling, directive advice, customer IDs |
| `invented_reason_codes` | `reasons_used` ⊆ the row's codes |
| `reason_coverage` | every reason code is mentioned (keywords match at a word start) |
| `impact_label` | an impact figure used in the text must be called an estimate |
| `insufficient_impact` | no emissions figure or claim when impact is `INSUFFICIENT_DATA` |
| `length` | headline ≤12 words, text ≤80 words |
| `low_certainty_hedge` (soft) | low certainty should use "may / could / might" (recorded only) |

Any hard failure leads to one retry with the failures fed back, then the template. Each explanation's `provenance` records:
- the text source (LLM / TEMPLATE)
- the provider and model, and the prompt version
- every attempt and its failures
- withheld evidence
- validation metrics
- a timestamp

**A real catch from the eval:** Gemini wrote "Based on **our model**, an upgrade could…". The validator rejected it, and the retry produced clean text.

## 7. Evaluation

### Eval set ([build_eval_set.py](../src/llm/eval/build_eval_set.py))

96 synthetic final rows in the engine's format: 4 products × 24 cases, seed 26, built from the catalogue and the team's reason-code dictionary. **Evidence is generated from each reason code**, so evidence always supports the code. (In an earlier version a case paired "recent home improvement spending" with £0, and the LLM dutifully wrote "you have spent £0".)

| Scenario | What it tests |
|---|---|
| normal | standard cases |
| low confidence | tentative wording |
| insufficient impact | no emissions figure stated |
| single reason / all reasons | coverage at both extremes |
| suppressed | no text produced |
| missing evidence | no evidence values available |
| decimal impact | small figures such as 2.9 kg |
| prompt injection | instructions in evidence fields |

The split is stratified: 48 dev / 48 test, with every product × scenario in both halves. Prompts are tuned on dev; the reported numbers are on test.

### Metrics ([run_eval.py](../src/llm/eval/run_eval.py), [judge.py](../src/llm/eval/judge.py))

- **First raw LLM attempt** (the model before guardrails): pass rate, numeric faithfulness, banned-phrase rate, reason coverage, impact-label correctness, low-certainty hedge rate.
- **What is shown** (the whole system): LLM-text rate, template-fallback rate, API-unavailable rate, retry rate, injection-leak rate, Flesch reading ease, length, latency.
- **Judge 1–5 rubric** on the card as seen: faithfulness, clarity, personalisation, tone, actionability, plus the % of cases with unsupported claims. Reported separately for LLM and template text.
- **Pairwise: LLM vs template** for the same row. Order is randomised by a case-ID hash. 1–5 scores hit the ceiling for both, while pairwise discriminates.

**Baseline:** the deterministic template. The LLM has to beat it head to head while matching it on every guardrail.

```bash
python -m src.llm.eval.build_eval_set
python -m src.llm.eval.run_eval --provider gemini --split test --judge
python -m src.llm.eval.run_eval --provider offline --split test --judge     # template baseline
```

### Results (test split, held out)

| Metric | Gemini (`gemini-3.7-flash`) | Template baseline |
|---|---|---|
| Cases / shown | 48 / 44 | 48 / 44 |
| Suppression accuracy | 1.00 | 1.00 |
| First-attempt pass rate (all hard checks) | **1.00** | 1.00 |
| Numeric faithfulness / banned phrases / reason coverage / impact labelling | 1.00 / 0 / 1.00 / 1.00 | 1.00 / 0 / 1.00 / 1.00 |
| Low-certainty hedging | 1.00 | 1.00 |
| Shown as LLM text / retries / API unavailable | 1.00 / 0 / 0 | — |
| Injection leaks | 0 | 0 |
| Flesch reading ease / words | 59.1 / 48 | 61.6 / 53 |
| Judge faithfulness, clarity, personalisation, tone, actionability (1–5) | 5.0, 4.98, 5.0, 4.45, 4.34 | 5.0, 4.79, 4.77, 4.81, 4.09 |
| Cases with unsupported claims (judge) | 0% | 0% |
| **Pairwise: preferred by the judge** | **89%** (39/44) | 11% |
| Latency per explanation | ~18.6 s (so explanations are pre-generated) | 0 |

Reading the numbers: the guardrail metrics are perfect for both by construction, so the comparison rests on the judge. The LLM wins head to head mainly on personalisation and clarity. The template scores higher on tone, and part of the margin comes from the template reading awkwardly with the team's evidence fields (e.g. "has vehicle loan yes"). The judge is an LLM (Gemma 4), so treat this as relative evidence.

Source: `reports/llm_eval/gemini_test.json`, `offline_test.json`. Cached responses in `data/llm_cache/`, so the numbers can be reproduced without a key.

Pre-port results on the earlier eval set (same prompt v2, same validator, older reason codes):

| Metric | Gemini | Template |
|---|---|---|
| Suppression accuracy | 1.00 | 1.00 |
| First-attempt pass rate | 0.955 (one "our model", one missed reason; both fixed on retry) | 1.00 |
| Numeric faithfulness / injection leaks | 1.00 / 0 | 1.00 / 0 |
| Shown as LLM text | 1.00 | — |
| Judge faithfulness / clarity / tone / actionability | 5.0 / 5.0 / 4.6 / 4.3 | 5.0 / 5.0 / 4.5 / 4.5 |
| **Pairwise: LLM preferred** | **71%** (n=42) | 29% |

### Integrated set: real deployed rows ([build_integrated_set.py](../src/llm/eval/build_integrated_set.py))

The synthetic set above tuned the prompt before integration. The reason codes, evidence
keys and impact methods have changed since, so those scores do not prove the current
contract. The integrated set takes rows straight from P5's `RecommendationService` over a
team bundle (default: the committed trained preview). The UI sends exactly these rows.
Selection is deterministic: per product, 4 rows with an estimated impact, 2 without one,
2 low-confidence rows, 1 row with data-quality flags, 2 suppressed rows (which must get no
text) and 1 row with a prompt injection added as free-text evidence. Some pools are smaller
than the quota: the v3 preview gives 40 cases (only 2 low-confidence rows), the earlier v2
preview gave 45.

```bash
python -m src.llm.eval.build_integrated_set            # -> data/llm_eval/integrated_set.jsonl
python -m src.llm.eval.run_eval --provider offline --set integrated          # template baseline
python -m src.llm.eval.run_eval --provider gemini --set integrated --judge   # needs GEMINI_API_KEY
```

Template baseline on the integrated set (`reports/llm_eval/offline_integrated.json`, 2026-10-11):

| Metric | v3 preview | v2 preview |
|---|---|---|
| Cases / shown / suppressed correctly | 40 / 32 / 1.00 | 45 / 37 / 1.00 |
| Pass rate (all hard checks) | 1.00 | 1.00 (0.97 before the fix below) |
| Numeric faithfulness / banned phrases / reason coverage / impact labelling | 1.00 / 0 / 1.00 / 1.00 | 1.00 / 0 / 1.00 / 1.00 |
| Low-certainty hedging / injection leaks | 1.00 / 0 | 1.00 / 0 |
| Flesch reading ease / words | 53.1 / 51 | 52.9 / 51 |

The first run found a real bug. A low-confidence Green Home card with three reasons,
two key figures and a two-part impact (kg and kWh) came to 81 words, one over the limit.
The template failed its own check and P5 fell back to its generic text. The template now
drops key figures first (never reasons, which must all stay covered) until the card fits.
A test now runs the template over every real integrated row. The same run showed that
the Green Home topic said "making your home more energy efficient" while the estimate is
for a heat-pump switch. The payload now names the modelled intervention.

**Live Gemini on the integrated set** (`gemini-3.7-flash`, 2026-10-11, v3 preview, 40 cases, 32 shown),
judged by two independent model families: Gemma 4 (Google) and `openai/gpt-oss-120b` on Groq.

| Metric | Gemini | Template |
|---|---|---|
| Suppression accuracy | 1.00 | 1.00 |
| First-attempt pass rate (all hard checks) | **1.00** | 1.00 |
| Numeric faithfulness / banned phrases / reason coverage / impact labelling | 1.00 / 0 / 1.00 / 1.00 | 1.00 / 0 / 1.00 / 1.00 |
| Shown as LLM text / retries / API unavailable / injection leaks | 1.00 / 0 / 0 / 0 | n/a |
| Flesch reading ease / words | 61.9 / 48 | 53.1 / 51 |
| Latency per explanation (free tier, so explanations are pre-generated and cached) | 15.9 s | 0 |
| Judge Gemma 4: faithfulness, clarity, personalisation, tone, actionability | 5.0, 5.0, 5.0, 4.59, 4.22 | n/a |
| Judge gpt-oss-120b: faithfulness, clarity, personalisation, tone, actionability | 4.97, 4.88, 4.94, 4.94, 4.34 | 5.0, 4.44, 4.41, 4.91, 3.84 |
| Cards with a claim the judge called unsupported (gpt-oss-120b) | 1 of 32 | 2 of 32 |
| **Pairwise: LLM preferred over the template** | **74%** (Gemma, n=31), **84%** (gpt-oss, n=32) | |

The one flagged Gemini card said a customer "already banks digitally" with 4.6 app logins a month;
both numbers are in the record, so the judge objected to the wording rather than to invented
content. Both judges prefer the LLM text mainly for clarity, personalisation and actionability.
They are LLMs, so this is relative evidence, not ground truth. Reports:
`reports/llm_eval/gemini_integrated*.json`, `offline_integrated*.json`; the responses are in the
committed cache, so the run replays offline.

```bash
python -m src.llm.eval.run_eval --provider gemini --set integrated --judge                         # Gemma judge
python -m src.llm.eval.run_eval --provider gemini --set integrated --judge --judge-provider groq   # gpt-oss judge
```

## 8. Known limitations

- The keyword-based reason-coverage check can pass a text that mentions the keyword without really explaining it. The judge is the backstop.
- The number check is exact-match, so correct paraphrases ("nearly two tonnes") are rejected. This is strict by design.
- The judge is itself an LLM (Gemma 4). Scores are relative evidence, not ground truth. A second judge family (Llama via Groq) is planned.
- On the free tier latency is about 18 s per explanation, so explanations are pre-generated (`python -m src.llm.explain <bundle>`) and served from the cache.
- The eval set is synthetic and English-only. Tone was not tested with real customers.
