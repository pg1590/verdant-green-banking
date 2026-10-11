# Team integration boundary

Run P5 with mocks immediately. Replace them using GREEN_BUNDLE_PATH, or
`python3 run_all.py --bundle path/to/team_bundle.json --serve app`.
Invalid team files stop loading; they never fall back to mock data.

## JSON bundle v1

```json
{
  "schema_version": "1.0",
  "data_mode": "TEAM_OUTPUTS",
  "customers": [{
    "customer_id": "C10001", "age": 38, "region": "Yorkshire and The Humber", "income_band": "Q3",
    "home_status": "owner_mortgage", "property_type": "semi_detached",
    "epc_rating": "E", "has_solar": false, "vehicle_type": "petrol",
    "statement_pref": "paper", "marketing_consent": true,
    "feature_cutoff_date": "2026-06-30", "data_completeness": 0.95,
    "conflict_flags": [], "data_flags": ""
  }],
  "predictions": {
    "C10001": {
      "GP01": {
        "propensity": 0.84, "confidence": 0.87,
        "reason_codes": ["HIGH_FUEL_SPEND"],
        "evidence": {"avg_monthly_fuel_spend_gbp": 210},
        "model_version": "gp01_lgbm_v1"
      }
    }
  },
  "impacts": {"C10001": {"GP01": {"status": "INSUFFICIENT_DATA"}}},
  "portfolio_history": []
}
```

This example leaves other model outputs absent deliberately; those products are
suppressed with MISSING_OR_INVALID_MODEL_OUTPUT. Production integration needs
P3 predictions for all GP01–GP04. Missing impact does not block an otherwise
valid display opportunity, but the UI must show insufficient data.

Customer IDs are unique. Enum strings are case-sensitive and canonical (see
`src/schemas/tables.py`; product rules are in `config/products.yaml`). Map source-specific spellings in P1's export, rather than guessing
ownership or suitability. NumPy scalars must become Python primitives. Use None
for missing values and standard JSON null; NaN/Infinity are rejected.

ESTIMATED impacts require at least one nonnegative finite annual metric,
nonempty method and assumptions, positive horizon_years, and a nonempty
`factor_edition` (the configured emissions-factor release). P2 supplies
`kg_co2e_per_year`, `energy_kwh_per_year`, `paper_sheets_per_year` as relevant.
P5 preserves `factor_edition` in recommendation JSON and shows it in the UI.
The cleaner fails closed to INSUFFICIENT_DATA for invalid estimates. TEAM_OUTPUTS
raises an explicit error if a raw ESTIMATED object is downgraded, including for
missing edition. Mixed emissions editions or a nonconfigured edition also raise
an error. P2 must mark genuinely missing inputs INSUFFICIENT_DATA. Even an
ESTIMATED object with only paper/energy metrics requires the release tag. Different conversion-price periods and assumption sources are
not emissions editions: P2 documents them independently in method/assumptions.
Unsourced or invalid impact is unavailable rather than guessed.

History is optional: a list of comparable snapshot objects with `date` (YYYY-MM-DD)
and `customers_with_opportunities` (integer). Empty history produces a clear
single-snapshot message. No trend is fabricated from cross-sectional data.

## P4 adapter

Construct `RecommendationService(bundle, insight_provider=p4_validated_explain)`.
The callable receives one final eligible recommendation dict and returns
`{"headline": "...", "text": "..."}`. Both fields must be nonempty strings.
The service preserves both fields and the UI displays the headline above the text.
Earlier nonempty string outputs remain supported with the product name as headline.
P4 owns validation, numeric faithfulness, retries and fallback before returning
that object. The action label stays in the recommendation's action field for the
button. P5's source label indicates the adapter contract, not an independent
faithfulness evaluation. API get_service() and the UI service() are the two small
composition points to install the same provider. Missing provider uses a simple
non-numeric deterministic template. Suppressed rows are never sent to the provider.
Each insight includes `headline`, `text`, `source`, `fallback_reason` and its
structured recommendation. Valid provider output has `fallback_reason=null`;
missing/invalid output is marked `MISSING_PROVIDER`/`INVALID_PROVIDER_OUTPUT`.
An invalid dict triggers a visible dashboard warning rather than silent replacement.

## API

- GET /health: mode and policy version
- GET /customers: synthetic customer snapshots for selection
- GET /customers/{id}/recommendations: eligible rows by default
- GET /customers/{id}/recommendations?include_suppressed=true: full audit
- GET /customers/{id}/insight: eligible explanations plus evidence and impact
- GET /portfolio/summary: opportunities, impact coverage and optional history
- GET /segments: ranked regional/product cohorts with drill-down customer IDs

A nonexistent customer returns 404. Server data is loaded once; restart after
replacing a bundle. Streamlit also caches by path; restart after modifying a file.
Do not use stale cached data to represent newly changed customer circumstances.

## P5 resolution of Sajiv's questions (2026-10-10)

Use `src.common.records.to_records()` when exporting pandas customer snapshots.
TEAM_OUTPUTS requires finite data_completeness in [0,1] and ISO
feature_cutoff_date. If data_as_of is supplied, it must describe the same cutoff
snapshot. Canonical non-null enums, known conflict_flags and known comma-separated
data_flags are checked at load. Both keys are required; export [] and an empty
string explicitly when there are no flags. Missing nullable attributes remain allowed and
are handled by product suppression. Labels, label windows, true_category and
hidden persona traits are rejected from customer bundles.

P1's existing `data_flags` string is retained; do not introduce a second quality
flag input. P5 passes it through into each recommendation. P3 accounts for quality
in its documented confidence method; P5 preserves model_confidence and caps at
completeness without applying another quality penalty.

`structurally_eligible` uses purpose='training' (consent excluded).
`eligible_to_show` includes display consent and valid model/evidence requirements.
For example a homeowner without consent can be structurally eligible for GP03
while eligible_to_show is false. In tests/prediction exports, state which scope
is evaluated; join training eligibility for eligible-cohort metrics.

Portfolio CO2e and energy use the highest-ranked eligible finance opportunity
per customer, without falling through to a lower-ranked product if impact is
missing. Paperless cannot displace that choice. GP04 paper savings come from all
eligible estimated paperless rows; its CO2e is reported under digital_impact
and excluded from the finance total. All totals are annual unweighted potential,
with coverage, not observed conversion or realised savings. Existing
potential_impact.paper_sheets_per_year remains a compatibility field for the UI.

See `docs/integration.md` for the decisions and each owner's next action.

impact_coverage_customers counts customers with finance CO2e estimates;
finance_estimate_customers counts customers with any estimated finance metric.
Use each metric’s own coverage_customers when reporting its total.

## Integrated main branch

The role branches are combined into one pipeline. P1’s `src/schemas/tables.py`
is the persisted raw/feature contract. P3 encodes its categorical model inputs in
`src/models/data.py`; these encodings do not rename the feature store. Age and
hidden traits/persona are excluded from model features. Percentile reason thresholds
are fitted per product on its training cohort and saved with the model; validation
and inference reuse them. P3 ships, per product, whichever of monotone LightGBM or
logistic regression scores higher in train CV; `model_version` names the family
(e.g. `gp01_logreg_v3`) and bundles live in `models/gp0X_propensity_v3.*` (see p3.md). Energy reasons use derived kWh, including apportioned
dual-fuel bills, rather than only single-fuel category spend.

Run `python scripts/run_pipeline.py --fast` to generate, categorise, build features,
train/calibrate, predict, compute sourced impacts and export the validated team
bundle before P4 explanations. The pipeline copies labels/splits into processed
paths for P3 and passes P2’s predicted category/charging/dual-fuel metadata to P1.
Historical tariffs and pre-July consumption assumptions are shared by both roles.
Conversion uses day-weighted window-average rates, an approximation to actual
customer tariffs. P1’s store uses the fixed feature window; the standalone P2
helper also retains an explicit observed-month annualisation mode.

API and Streamlit call `src.recommend.runtime.load_runtime_service()` so P4’s
provider and customer memory are installed in both. The generated bundle is the
runtime default; an explicit GREEN_BUNDLE_PATH takes precedence and is validated.
An absent generated bundle uses the committed labelled trained preview. If neither
exists, fixtures require explicit GREEN_ALLOW_DEMO=1. GREEN_MEMORY_PATH can override
the local SQLite file; default feedback databases are namespaced by mode and bundle
fingerprint, so rebuilding synthetic identities does not inherit old feedback.
Memory never changes structural eligibility, model scores or impact. Customer
feedback can hide/cool off a valid offer without changing eligible_to_show in the
audit; its separate memory metadata records the display decision.

Additional endpoints:
- POST `/customers/{id}/events` with `{"product_code":"GP01","event":"dismissed"}`.
  Events: shown, clicked, dismissed, not_relevant, took_up.
- DELETE `/customers/{id}/memory` clears stored feedback.
- GET `/segments/insight?region=London&product_code=GP01` returns P4’s validated
  business explanation for an eligible segment.

The held-out evaluator writes exactly the eight canonical test CSV columns;
eligible_to_show is P5’s final display decision before optional feedback memory.
Train/validation/test metrics use structural eligibility, without consent filtering.
Normal pipeline runs do not evaluate the test set; use the separate one-shot
command only after models are frozen.

Carbon integration estimates all four products from one documented scenario each
(D22, `docs/carbon_methodology.md`): ICE-to-BEV, gas boiler to heat pump, a
reference 3.5 kWp PV design and paper statements as a lower bound. Missing or
unsupported inputs remain INSUFFICIENT_DATA with a reason code and never become zero.

## Trend, opportunities and Customer 360 (2026-10-11)

- `portfolio_history` is built by `src/features/snapshots.py` (pipeline stage before
  the bundle export) and written to `data/processed/portfolio_history.json`. Each
  snapshot keeps the required `date` and `customers_with_opportunities`, and adds
  `eligible_opportunities`, `opportunities_by_product`, `kg_co2e_per_year`,
  `customers_analysed`, `window_start`, `window_end`, `model_versions` and
  `series` (`RETROSPECTIVE_RESCORING_FIXED_COHORT`). The validator still checks only
  the required fields, so older bundles load unchanged.
- GET `/portfolio/opportunities` returns `RecommendationService.opportunities()`:
  per product, the structurally eligible cohort, displayed opportunities (same count
  as the summary), high-priority count and threshold (D24), median and P90
  propensity, expected take-ups and impact coverage, with written definitions.
- Customer records may carry P1's Customer 360 display fields: `products_held`
  (canonical product types), `n_transactions_window`, `active_months_window`
  (nonnegative integers) and `last_transaction_date`. They are validated when present
  and are never model inputs.
- `recommendations(..., on=date)` sets the memory date for replaying dated demo
  sessions; the default is today.

## New in the stress-test round (2026-10-11)

- **Unavailable impacts keep their reason.** `clean_impact` now passes P2's `reason` code through
  (for example `MISSING_GAS_USE`); `src.carbon.engine.UNAVAILABLE_REASONS` gives the plain-English text
  the UI shows. Valid estimates are unchanged.
- **Portfolio views are cached.** `summary()`, `opportunities()`, `segments()`, `optional_summary()`
  and `engagement()` are computed once and reused until the date or the feedback store changes
  (`MemoryStore.stamp()` also sees writes from another process). The API warms them at startup.
  Callers get copies, so mutating a result cannot corrupt the cache.
- **Optional products** (Green Investments GP05, Green Loans GP06; slide 4) live in
  `config/optional_products.yaml` and `src/recommend/optional.py`. They are rule-based signals with
  evidence, have **no propensity** (there are no synthetic take-up labels) and impact stays
  INSUFFICIENT_DATA with a reason. They are never mixed into the ranked GP01-GP04 list.
  GET `/customers/{id}/optional`, GET `/portfolio/optional`.
- **What-if:** POST `/customers/{id}/what-if` with `{"changes": {"vehicle_type": "none"}}`.
  Only profile facts in `src.recommend.what_if.EDITABLE` can change. Rules, P3's frozen models and
  P2's engine run again; the response gives before/after per product and is never saved.
- **Ask the portfolio:** POST `/portfolio/ask` with `{"question": "..."}`. The LLM picks one query
  from a fixed menu, the service computes the facts, and the answer is checked against them
  (`src/llm/ask.py`). Without a provider a keyword router and a template answer are used.
- **Engagement funnel:** GET `/portfolio/engagement` gives distinct customers shown, clicked,
  dismissed, not relevant and took up per product, from the feedback memory (demo interactions,
  not bank conversions).
