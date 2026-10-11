# Decision log

Contract and design decisions. **Agreed** means the team has confirmed it. **Proposed** is the default the scaffold uses, and the team should confirm or change it during the contract freeze (H0–1.5). Changing a decision after the freeze needs P5's approval, plus an update to this file and the matching config or schema.

| ID | Decision | Status | Where it lives |
|---|---|---|---|
| D1 | Labels depend on hidden traits and noise that never appear as features | **Agreed** 2026-10-10 | `src/schemas/tables.py` `LATENT_TRAITS`, `tests/test_contracts.py` |
| D2 | Calendar: 12-month feature window, 3-month label window | Proposed | `config/settings.yaml` |
| D3 | Profile attributes and products held are taken as of the feature cutoff | Proposed | `src/features/build.py` |
| D4 | Customer-level 70/15/15 split for propensity; merchant-level split for FR2 | Proposed | `src/data_gen/splits.py`, `src/categorisation/model.py` |
| D5 | Recency features detect changed circumstances | Proposed | `FEATURES` in `src/schemas/tables.py` |
| D6 | Dual-fuel bills stay in Utilities; features apportion them to electricity and gas | Proposed | `config/categories.yaml`, `src/features/build.py` |
| D7 | DESNZ 2026 emissions factors everywhere | Proposed | `config/settings.yaml`, `config/emission_factors.csv` |
| D8 | Propensity models train on the eligible population only | **P5 filter implemented; P1/P3 pipeline wiring pending** | `eligible_customer_product_pairs()` in `src/recommend/eligibility.py` |
| D9 | Age is excluded from model inputs and used only for fairness checks | Proposed | `config/settings.yaml` `model.excluded_features` |
| D10 | The LLM layer works without an API key (committed cache plus template fallback) | **Implemented by P4** | `src/llm/`, `data/llm_cache/` |
| D11 | Digital adoption is reported separately from CO₂e | **Implemented by P5** (one finance intervention per customer; paperless impact separate) | `src/recommend/service.py`, dashboard |
| D12 | Streamlit and FastAPI share one service | **Implemented by P5** | `RecommendationService` in `src/recommend/service.py` |
| D13 | 10,000 customers, parquet files, generated data not committed | **Decided by P1** 2026-10-10 | `config/settings.yaml`, `.gitignore` |
| D14 | ~~P2 owns the catalogue~~ P5 owns the catalogue and eligibility rules | **Withdrawn**: P5 shipped them | `config/products.yaml` |
| D15 | Format of `test_predictions.csv` | Proposed | `TEST_PREDICTION_COLUMNS` |
| D16 | Paperless impact covers paper only; postage is excluded unless a sourced factor is found | Proposed | `src/carbon/engine.py` |
| D17 | Taxis and ride-hailing count as Other, not Public Transport | Proposed | `config/categories.yaml` |
| D18 | No marketing consent suppresses **finance** products (GP01–GP03) for display; training ignores consent | **Implemented by P5** | `check_eligibility(purpose=...)` |
| D19 | P5's integration contracts are canonical (catalogue, eligibility, recommendation row, impact fields, bundle) | **Agreed** at merge, 2026-10-10 | `docs/integration.md`, `docs/integration.md` |
| D20 | Feature store v1 definitions and thresholds (historical Ofgem prices) | **Decided by P1** 2026-10-10 | `docs/data.md`, `config/settings.yaml` `features:` |
| D21 | Label rules v1: true-state drivers spread across several signals + hidden traits + noise, target rates, ruled-out customers get 0 | **Decided by P1** 2026-10-10 (P3 reviews) | `config/label_rules.yaml`, `docs/data.md` |
| D22 | One documented impact scenario per product: heat-pump switch (GP02), reference 3.5 kWp PV with MCS self-consumption and no export credit (GP03), one sheet per statement as a lower bound (GP04) | Proposed 2026-10-11 (P2 to confirm) | `src/carbon/engine.py`, `docs/carbon_methodology.md`, `config/solar_self_consumption.csv` |
| D23 | Trend = month-end snapshots 2026-06-30 to 2026-09-30, each with its own 12-month window, fixed cohort, rescored by the frozen current models | Proposed 2026-10-11 (P1/P3/P5 to confirm) | `src/features/snapshots.py`, `docs/data.md` |
| D24 | High-priority opportunity = propensity at least 1.5x the product's mean eligible-cohort propensity; expected take-ups = sum of calibrated propensities | Proposed 2026-10-11 (P5 to confirm) | `RecommendationService.opportunities()` |
| D25 | Customer 360 export: feature row + products held at cutoff + feature-window activity counts, one row per customer; display only, never model inputs | Proposed 2026-10-11 (P1/P5 to confirm) | `src/features/build.py` `customer_360()`, `data/processed/customer_360.parquet` |
| D26 | Optional products GP05/GP06 are rule-based signals with evidence, no propensity, no invented impact | Proposed 2026-10-11 | `config/optional_products.yaml`, `src/recommend/optional.py` |
| D27 | What-if replays profile edits through rules, frozen models and the engine; transaction features stay as observed; never saved | Proposed 2026-10-11 | `src/recommend/what_if.py` |
| D28 | Ask the portfolio: the LLM routes to a fixed query and phrases computed facts; numbers checked | Proposed 2026-10-11 | `src/llm/ask.py` |
| D29 | Bill-derived gas use above twice Ofgem's high benchmark is not used for a heat-pump estimate | Proposed 2026-10-11 | `src/carbon/engine.py` |

## Details

**D1: Hidden traits and noise.** The generator gives every customer hidden traits: `eco_mindedness`, `digital_affinity`, `financial_headroom` and `change_readiness`. These shape both behaviour and take-up, and take-up also gets random noise. The model never sees the traits, so held-out metrics stay realistic and don't simply recover the generator. The traits are written to `data/raw/latent_traits.parquet` for diagnostics only. One diagnostic worth showing judges is qualitative agreement between SHAP explanations and documented observable label relationships. Do not claim exact recovery of the generator’s β values: hidden traits, correlations and model/calibration transformations affect attributions.

**D2: Calendar.** Transactions run from 2025-07-01 to 2026-09-30. Features use 2025-07-01 to 2026-06-30, so `feature_cutoff_date` and `data_timestamp` are both 2026-06-30. Labels come from take-up between 2026-07-01 and 2026-09-30. The label window ends before the hackathon date, which keeps the setup realistic.

**D3: As-of snapshots.** `customers` describes each customer as they were on 2026-06-30. A customer who took EV finance in August 2026 still shows their petrol car, and products opened after the cutoff are ignored. Without this, label information leaks into the features.

**D4: Splits.** Propensity models use a customer-level split, so every customer sits in exactly one of train, validation or test. The test set is scored once, at the end. FR2 is split by **merchant brand**, so the categoriser is tested on merchants it never saw. A transaction-level split would give near-perfect, meaningless scores.

**D5: Changed circumstances.** The feature store includes `spend_fuel_recent_m` (last 3 months), `spend_fuel_prior_m` (the 9 months before) and `months_since_last_fuel_txn`. A sharp drop sets `RECENT_CIRCUMSTANCE_CHANGE` in the existing comma-separated `data_flags`. P3 accounts for this in its documented confidence method; P5 passes it through without a second penalty. Confirmed vehicle loss uses `VEHICLE_SOLD` in `conflict_flags` to suppress EV finance. Low spend alone is not evidence that a vehicle was sold.

**D6: Dual-fuel bills.** FR2 requires exactly one category per transaction, and a combined gas-and-electricity bill is not a single-fuel bill, so it goes in Utilities. The feature layer then splits the amount into estimated electricity and gas use:
1. Expected costs use Ofgem medium consumption (2,500 kWh electricity, 9,500 kWh gas) multiplied by the price-cap unit rates.
2. The bill is divided between electricity and gas in proportion to those costs.

Customers handled this way get `ENERGY_APPORTIONED_FROM_DUAL_FUEL` in `data_flags`; P3 owns the confidence adjustment. P1/P2 must use consumption assumptions and price periods appropriate to the feature cutoff. The scaffold currently has Jul–Sep 2026 prices and TDCVs (after the 2026-06-30 cutoff): source earlier values before using these derived quantities in model features, or leave them unavailable.

**D7: Emissions factors.** Every factor comes from the DESNZ 2026 edition (published 11 June 2026). The 2026 grid electricity factor is much lower than 2025, so mixing editions would distort the solar and EV impacts. Every ESTIMATED impact must carry a nonempty `factor_edition`; INSUFFICIENT_DATA uses null. P5 preserves it, exposes it in summary/UI, and rejects a bundle mixing emissions editions. Prices and assumptions retain their own source-period provenance. P2 owns validation of the numerical factors and revisions.

**D8: Training population.** Each product's model trains only on customers the rules allow it to be shown to. Otherwise the model learns, for example, that EV owners score 0, which would be the model doing the eligibility rules' job. P1 must use the same training-purpose eligibility policy when constraining labels, and P3 must call the helper when filtering cohorts. Their pipeline wiring is pending; implementing the P5 helper alone does not complete training. Consent is excluded from this structural cohort.

**D9: Age.** Age is a protected characteristic under the Equality Act 2010. It stays out of model inputs but is kept in the feature table so the disparity check (propensity by age band, region and income) can run. Income band remains a propensity-model input; it does not establish affordability or credit approval. P3/P4 document proxy risks and subgroup performance. Age also remains available to the explicit adult eligibility gate, independently of ML.

**D10: Running without an API key.** LLM responses are cached by a hash of provider, primary model, prompt text and the minimised payload, and the cache is committed to the repo. Temperature is left at the Gemini default, because the cache, not sampling settings, is what makes reruns identical. With no key and no cache hit, the deterministic template is used. Judges can therefore rerun everything offline (docs/llm_design.md).

**D11: Digital adoption.** The speaker notes for slide 14 say to report digital adoption separately from environmental impact. The portfolio headline uses each customer’s highest-ranked eligible finance intervention for CO₂e/energy; there is no fallback to a lower-ranked estimate when impact is missing. GP04 sheets, digital adoption opportunities and optional paperless CO₂e are separate metrics. GP04 cannot displace an EV/home/solar opportunity from the finance total. Report estimate coverage; none of these are realised savings.

**D12: UI and API.** P5's `RecommendationService` loads either labelled demo fixtures or an explicit team bundle (`GREEN_BUNDLE_PATH`), and both Streamlit and FastAPI use it. An invalid team bundle fails loudly; it never falls back to mock data.

**D13: Data size.** 10,000 customers (plus 13 scenario customers) over 15 months gives about 4.86M transactions, about 58 MB as parquet. The full seeded pipeline rebuilds in about 5 minutes on a laptop, so judges can rerun it. Generated data is rebuilt by the seeded pipeline rather than committed. Committed: the 200-customer sample, models, LLM cache, the 213-customer preview and outputs.

**D14: Ownership (withdrawn).** P5 published the catalogue and executable eligibility rules before the merge, so P5 keeps them. Policy changes go through P5.

**D15: Test predictions.** `outputs/test_predictions.csv` has one row per test customer and product, with columns `customer_id, product_code, propensity, confidence, reason_codes, eligible_to_show, took_up, model_version`.

**D16: Postage.** We have not found a postage or letter-delivery factor in the DESNZ set (P2 to double-check the freight tables). Following the brief's instruction not to invent a method, paperless impact covers paper production only, unless the team finds and cites a documented factor.

**D17: Taxis.** Taxis are private road transport, so they count as Other rather than Public Transport. Rail, bus, coach, tram, underground and ferry count as Public Transport.

**D18: Marketing consent.** As implemented by P5: missing or false consent suppresses finance products (GP01–GP03) for display. Paperless is a service, so it can still show. `purpose='training'` ignores consent, so consent never shapes the training population.

**D19: Merge of the two scaffolds (2026-10-10).** P5's code is canonical wherever the scaffolds overlapped. In practice:
- **GP04's opportunity code is `PAPERLESS_BANKING`.**
- **The recommendation is the dict row from `src/recommend/engine.py`.** The earlier pydantic model was dropped.
- **Impact metrics are `kg_co2e_per_year`, `energy_kwh_per_year` and `paper_sheets_per_year`.**
- **An EV owner needs explicit `ev_replacement_use_case=True`; `VEHICLE_FUEL_CONFLICT` suppresses GP01 even when replacement is intended.** It does not use a fuel-spend threshold, because EV charging is also in the Fuel category.
- **Green Home needs EPC D–G or an explicit use case.**
- **Flats get `PROPERTY_REQUIRES_SUITABILITY_REVIEW`.**

The non-P5 contracts (raw tables, feature store, reason codes, factors, categories, calendar) come from the other scaffold, extended with the fields P5's rules read (`conflict_flags` and the use-case flags).

## P5 response — resolved 2026-10-10

The three open questions are answered in [integration.md](integration.md):

1. **Yes:** preserve `factor_edition` as its own field, required for ESTIMATED impact.
2. **Use the canonical schema:** `Yorkshire and The Humber`, `Q1`–`Q5`; demo middle-income fixture is `Q3`.
3. **Adopt P1's names:** `spend_fuel_recent_m` and `spend_fuel_prior_m`, both GBP/month averages. Retire `fuel_spend_last3m`; do not alias sums and averages.

This change also corrects structural/display eligibility separation and finance/
paperless portfolio aggregation. Other owners' unfinished pipelines remain pending.

**D20: Feature store v0.** The definitions are in `docs/data.md`. Points to confirm:
- the £40/month road-fuel conflict threshold
- the 30-day month approximation
- the formula for `data_completeness`
- leaving the use-case flags null

**D21: Label rules v0.2.** The details and the learnability results are in `docs/data.md`. The open choice is between 5,000 and 10,000 customers, and how much to spread label weight across drivers. Decide on realism grounds, not to make the model win.

**Open items from building v0 (for P3 and P5):**
1. **P5's engine suppresses any row with no reason codes** (`MISSING_OR_INVALID_REASON_CODES`). With evidence-gated codes, 2,795 of 20,000 mock rows are suppressed only because no evidence crossed a threshold. Is "no strong evidence, so no offer" the intended behaviour, or should P3 always emit at least the top SHAP code?
2. **`check_eligibility()` re-reads `products.yaml` on every call** unless given a `catalogue`. Pass `load_catalogue()` once when looping over rows; it made label generation about 25× faster.
3. **Use-case flags are null in v0,** so every EV owner and every customer without a vehicle is suppressed for EV Finance. If first-EV buyers should be reachable, P1 needs a rule for `ev_purchase_use_case`.

**P1 dataset decisions (2026-10-10).** The full structure is in the shared schema doc and `docs/data.md`. In short:
- **New columns:** `urban_rural`, `household_size`, `num_vehicles`; maisonette dropped (it counts as a flat).
- **New features:** `spend_total_m`, `fuel_share_of_spend`, `energy_spend_per_person_m`, `km_per_vehicle`.
- **Energy prices:** historical Ofgem cap prices per quarter (`config/energy_prices.csv`) and pre-July-2026 consumption values. This resolves P5's D5/D6 condition.
- **Traits and personas:** 4 hidden traits, partly linked to observable data; 6 hidden personas raked to national totals.
- **Labels:** weight spread across several drivers.
- **Size:** 10,000 customers.
- **Calibration:** national totals only (`config/calibration_targets.yaml`, `reports/calibration_report.md`).

**D22: Impact scenarios (2026-10-11).** EPC alone does not give a saving, and home ownership alone does not give a roof design. So each product gets one documented scenario applied to the customer's own data, and everything else stays INSUFFICIENT_DATA. Green Home: gas boiler to air-source heat pump (84% boiler efficiency, BEIS 2022; SPF 2.80, Electrification of Heat trial), only for gas-heated homes with known gas use. Solar: 3.5 kWp south-facing (DESNZ Solar Roadmap 2025), PVGIS regional yield, MCS MGD 003 "in half the day" self-consumption by generation and the customer's electricity use (1,500-5,999 kWh only), houses only, no export credit. Paperless: one sheet per statement, the minimum, so a lower bound; no estimate when no paper statements were sent. Overlapping home interventions are not added together because the portfolio uses one finance intervention per customer (D11). Details and worked examples: `docs/carbon_methodology.md`.

**D23: Trend snapshots (2026-10-11).** Data starts on 2025-07-01, so a full 12-month window only exists for month-ends from 2026-06-30 to 2026-09-30. Each snapshot rebuilds features from its own window with the same code, then the frozen current models rescore it and P5's service aggregates it. The cohort, rules and impact method are fixed. This is retrospective rescoring, not a history of deployed models. Post-June snapshots overlap the label window, so they are display-only and never used for model evaluation. The Jul-Sep 2026 Ofgem tariff was added to `config/energy_prices.csv` for these windows only.

**D24: Opportunity table (2026-10-11).** Base rates differ a lot (5% Solar to 27% Paperless), so a fixed propensity cut-off would mark nothing or everything. High priority is relative to each product's own eligible cohort. Expected take-ups are model-expected take-ups within the 3-month label window if every shown customer were contacted. They are not uplift and not observed conversions.

**D26: Optional products (2026-10-11).** The brief lists Green Investments ("existing investments, sustainability preference") and Green Loans ("existing borrowing, eligible use case") as optional. The synthetic data has no take-up labels for them, so training a model would mean inventing labels. They are therefore shown as rule-based signals: existing holdings or borrowing are required, plus at least one sustainability signal (EV, solar, public transport at twice the ONS average, EPC A/B) or an eligible use case (home EV charger for a single-car EV owner in a house who charges away from home; efficient appliances above Ofgem's high electricity benchmark). Green Loans exclude purchases covered by GP01-GP03. No sourced impact method exists (PCAF needs fund holdings; a home charger uses the same grid electricity), so impact stays unavailable with a reason.

**D27: What-if (2026-10-11).** Judges test "changed customer circumstances". A profile edit runs the same components again, each doing its own job. Spending, mileage, energy use and conflict flags come from transactions and are not changed by an edit, which the response states.

**D28: Ask the portfolio (2026-10-11).** The LLM may only choose one of seven fixed queries and word the result. Every number comes from the service, and an answer with a number not in the facts, or approval or guarantee language, is rejected and replaced by the facts themselves.

**D29: Gas plausibility (2026-10-11).** 73 of 10,013 customers had bill-derived gas use above 34,000 kWh (twice Ofgem's high TDCV of 17,000 kWh; maximum 71,064 kWh), most likely business use or misread bills in the synthetic generator. Those customers contributed 2.8% of the Green Home potential. Their heat-pump estimate is now unavailable (`GAS_USE_OUTSIDE_PLAUSIBLE_RANGE`). The feature itself is unchanged because the models were trained on it; P2 should review the conversion.
