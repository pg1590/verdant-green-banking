# Data

All customer-level data is **fully synthetic**. Public statistics in `DATA_SOURCES.md` are used only to calibrate distributions. No real personal or banking data is used.

## Tables

The authoritative schema is `src/schemas/tables.py`. Every stage validates its output against it with `validate_table`.

| Table | Grain | Notes |
|---|---|---|
| `customers` | One row per customer | Profile **as of** `feature_cutoff_date` (D3). Nullable fields carry the missing-data scenarios |
| `products_held` | Customer × product | `close_date` records changes such as a car being sold |
| `transactions` | One row per transaction | `true_category` is for FR2 evaluation only |
| `digital_activity` | Customer × month | App logins, web sessions, paper statements |
| `labels` | Customer × product | `took_up` in the label window |
| `splits` | One row per customer | train / val / test (70/15/15), customer-level (D4) |
| `latent_traits` | One row per customer | **Diagnostics only**; never a feature (D1) |

Feature store: one row per customer, columns defined by `FEATURES` in `src/schemas/tables.py`.

## Export to P5

P5's eligibility rules and integration bundle read customer snapshots (`docs/integration.md`, `docs/integration.md`). Requirements:
- **Export through `src.common.records.to_records()`.** It turns dates into ISO strings and NaN or NA into `None`. P5's engine rejects Timestamps, and the bundle rejects NaN.
- **`conflict_flags` is a list of codes** from `CONFLICT_FLAGS`. `VEHICLE_FUEL_CONFLICT` must come from **road-fuel** spend (pump merchants), never EV charging.
- **A car sold in the window** becomes `vehicle_type = "none"` plus `VEHICLE_SOLD`.
- **Optional use-case flags** (`ev_replacement_use_case`, `green_home_use_case`, …) switch on product paths that the rules otherwise suppress. Generate them only with a documented rule.
- **Enums use the schema's canonical values.** Regions are ONS names; income bands are `Q1`–`Q5`.

## Customer 360 export (FR1, D25)

`python -m src.features.build` also writes `data/processed/customer_360.parquet`: exactly
one row per customer with the full feature row plus:

| Field | Meaning |
|---|---|
| `products_held` | Sorted list of product types open at the cutoff (`open_date <= cutoff`, not closed by then) |
| `n_transactions_window` | Transactions dated inside the feature window (2025-07-01 to the cutoff) |
| `active_months_window` | Months in the window with at least one transaction |
| `last_transaction_date` | Last transaction date in the window (ISO), or null |

No post-cutoff activity enters the view. These are display fields only; models use the
explicit feature lists in `config/model.yaml`. P5's bundle export merges them into each
customer record, and the dashboard's Customer 360 panel shows them.

## Trend snapshots (D23)

`python -m src.features.snapshots` (a pipeline stage) rebuilds the feature store at four
month-ends, 2026-06-30, 2026-07-31, 2026-08-31 and 2026-09-30. Each one uses its own
12-month window ending on the snapshot date, the same raw tables and the same P2
categories. Earlier dates would need a shorter window, because data starts on 2025-07-01,
so they are not used. Every snapshot has the same customers, catalogue rules and impact
method. The frozen current models rescore each one, which makes the series retrospective
rescoring, not a record of models deployed at the time. Profile attributes are the
2026-06-30 snapshot. Later months see change only through transactions, digital activity
and dated holdings. The synthetic generator is stationary month to month, so a fairly
flat trend is the expected, honest result. Post-June snapshots overlap the label window
and are never used for model evaluation.

## Calendar (D2)

| | From | To |
|---|---|---|
| Transactions | 2025-07-01 | 2026-09-30 |
| Feature window | 2025-07-01 | 2026-06-30 (`feature_cutoff_date`) |
| Label window | 2026-07-01 | 2026-09-30 |

## Generation process (v1)

`src/data_gen/generate.py` (`make data`: 10,000 customers to `data/raw/` in about 1 minute; `make sample`: 200 to `data/sample/`). Seeded and reproducible. The agreed structure is in the shared schema doc.

1. **Persona:** each customer gets one of 6 personas (`config/personas.yaml`): commuter family, urban renter, retired owner, stretched household, rural owner, eco-minded professional. It's hidden.
2. **Attributes:**
    - Age band, household size, cars, urban/rural, property type, income, heating fuel, EPC and main vehicle type come from persona weights, **raked** (iterative proportional fitting) so the national mix equals `config/calibration_targets.yaml`.
    - Tenure is drawn from the real EHS tenure-by-age table.
    - Region comes from national shares.
    - Solar is possible only for owner-occupied houses, scaled to the national 6.1%.
3. **Hidden traits:** persona mean + N(0,1) noise, partly linked to income, age and urban living, then standardised.
4. **Spending:** depends on household size, property, EPC, heating fuel, cars, area and traits. National means per category are then scaled to ONS Family Spending FYE 2025.
5. **Record errors are injected after the truth snapshot:**
    - 0.5% sold their only car in the window
    - 0.33% recorded as EV while really driving a petrol or diesel car
    - missing EPC 8%, property type 3%, heating 5%, vehicle 3%, solar 3%
6. **Labels** come from the truth (below).

**Calibration result** (`reports/calibration_report.md`, `make calibration`):
- 58 share checks against official figures: mean gap 0.29 points, largest 1.4.
- Spend matches ONS within 2%.
- Implied kWh medians (2,672 electricity, 11,127 gas) independently match Ofgem's typical household (2,700 / 11,500).
- The sources disagree on mileage: ONS fuel spend implies about 5,300 miles per car, while the NTS says 7,100. We match ONS spend.

## Scenario customers (judge tests)

13 hand-designed customers with fixed IDs **C90001–C90013** (`config/scenarios.yaml`) are generated on top of the population with the same code, so their transactions are realistic. Each forces one situation from the brief: missing or conflicting data, products that must not be shown, insufficient impact data, changed circumstances. Each has an expected outcome per product.

- **Use:** P5's demo (real generated customers instead of hand-typed fixtures) and `tests/scenarios/test_pipeline_scenarios.py`, which checks every expected outcome end to end (generator → features → P5 rules) with no trained model needed.
- **Excluded** from labels, splits, training, evaluation and the calibration report. They're deliberate edge cases, not part of the population. They do appear in the feature table and the bundle, so they get scored and shown.
- **Untouched by random record errors:** they differ from the truth only where the scenario says so.

| ID | Scenario | Expected |
|---|---|---|
| C90001 | High-mileage commuter family (happy path) | All four products shown |
| C90002 | Missing EPC rating | Green Home: MISSING_HOME_UPGRADE_EVIDENCE |
| C90003 | Missing vehicle record | EV: MISSING_VEHICLE_TYPE |
| C90004 | EV owner, one car | EV: ALREADY_HAS_EV |
| C90005 | Record says EV, buys road fuel | EV: VEHICLE_FUEL_CONFLICT |
| C90006 | Sold the only car in March 2026 | EV: VEHICLE_SOLD; RECENT_CIRCUMSTANCE_CHANGE flag |
| C90007 | Renter, high electricity | Green Home and Solar: NOT_HOMEOWNER |
| C90008 | Already has solar | Solar: ALREADY_HAS_SOLAR |
| C90009 | Owns a flat | Solar: PROPERTY_REQUIRES_SUITABILITY_REVIEW; Green Home shown |
| C90010 | Already paperless | Paperless: ALREADY_DIGITAL |
| C90011 | Dual-fuel bill only | ENERGY_APPORTIONED_FROM_DUAL_FUEL flag; Green Home shown |
| C90012 | No marketing consent | EV, Green Home, Solar: NO_MARKETING_CONSENT; Paperless shown |
| C90013 | Two-car EV household | EV: ALREADY_HAS_EV; **no** fuel-conflict flag |

## Feature definitions (v1)

Built by `src/features/build.py` (`make features`). The window runs from 2025-07-01 to 2026-06-30: 12 months, of which the last 3 are "recent". Thresholds live in `config/settings.yaml` under `features:`.

| Feature | Definition |
|---|---|
| `spend_<category>_m` | Sum of **predicted** FR2 category spend in the window ÷ 12. For now the rules baseline (`src/categorisation/rules.py`) predicts the categories; P2's model replaces it later |
| `spend_fuel_recent_m` / `spend_fuel_prior_m` | Fuel-category spend per month over the last 3 months / over the 9 months before them |
| `months_since_last_fuel_txn` | Whole 30-day months from the last Fuel transaction to the cutoff; null if there was none |
| `est_annual_km` | For petrol, diesel and hybrid only: 12-month **road-fuel** spend (Fuel minus EV charging) ÷ mean pump price × fuel economy. Economy is derived from DESNZ factors: petrol 12.85, diesel 14.96, hybrid 16.01 km/litre. Null for EV, no vehicle, or a missing vehicle type |
| `est_electricity_kwh_year` | Annual electricity spend, plus the electricity share of dual-fuel bills, minus the window's standing charges, ÷ the day-weighted unit rate of the four Ofgem cap quarters Jul 2025–Jun 2026 (`config/energy_prices.csv`). Floored at 0; null if there's no electricity spend at all |
| `spend_total_m`, `fuel_share_of_spend` | Total monthly spend; fuel ÷ total |
| `energy_spend_per_person_m` | (Electricity + gas, including dual-fuel bills) ÷ household size, per month |
| `km_per_vehicle` | `est_annual_km` ÷ `num_vehicles` |
| `est_gas_kwh_year` | The same for gas. **0** when there's no gas spend and heating isn't gas; **null** when there's no gas spend but heating is gas or unknown |
| `energy_from_dual_fuel` | Any Utilities transaction matches the dual-fuel keywords (D6). Electricity's share of the bill is fixed: expected bill at Ofgem medium consumption (2,500 kWh electricity, 9,500 kWh gas) including standing charges, giving ≈ 0.525 (pre-July-2026 consumption values 2,700 / 11,500 kWh at feature-window prices) |
| `has_*` | Product open on the cutoff date: `open_date` ≤ cutoff and (`close_date` empty or after the cutoff) |
| `vehicle_loan_closed_in_window` | A vehicle loan closed during the window |
| `avg_app_logins_m`, `avg_web_sessions_m`, `paper_statements_12m` | Over the window's monthly activity rows |
| `data_completeness` | Mean of 8 parts: whether each of the 7 profile fields (home status, property type, EPC, heating fuel, solar, vehicle type, statement preference) is present, plus the share of months with any transaction |

**Conflict flags (P5 codes):**
- `VEHICLE_FUEL_CONFLICT`: EV on record, at most one car, and road-fuel spend of £40 or more per month. Two-car EV households legitimately buy fuel.
- `VEHICLE_SOLD`: no vehicle on record, and either road fuel stopped (at least £40/month before, under £5/month in the last 3 months) or a vehicle loan closed in the window.
- `VEHICLE_STATUS_CONFLICT`: a combustion car on record whose road fuel stopped, or no vehicle on record but £40 or more of recent road fuel.
- `HOME_STATUS_CONFLICT`: a renter with a mortgage.
- `STATEMENT_PREF_CONFLICT`: digital preference, but paper statements were sent in the last 3 months.
- `EPC_CONFLICT` and `SOLAR_STATUS_CONFLICT` aren't generated yet, because there's no second source of evidence to compare against.

**Data flags:**
- `LOW_DATA_COMPLETENESS`: completeness below 0.8.
- `CONFLICTING_DATA`: any conflict flag is set.
- `RECENT_CIRCUMSTANCE_CHANGE`: fuel spend was £40 or more per month before, and recent spend is below 25% of that.
- `ENERGY_APPORTIONED_FROM_DUAL_FUEL`.

**Use-case flags** (`ev_replacement_use_case` and the others) are left **null** in v0, because no generation rule has been agreed. As a result, P5's rules suppress EV Finance for every EV owner and every customer without a vehicle.

## Label rules (v1)

Rules are in `config/label_rules.yaml` and the code is in `src/data_gen/labels.py`: same formula as before, with weight spread across several drivers (decision 5). Drivers use the generator's true state and are standardised within each product's eligible group. Ruled-out customers always get 0.

| Product | Target | Observable drivers (coef) | Hidden traits (coef) |
|---|---|---|---|
| GP01 EV | 8% | fuel per car 0.6, fuel spend 0.3, car loan 0.6, 2+ cars 0.4, urban 0.3, income 0.3 | eco 0.6, headroom 0.4, change 0.3 |
| GP02 Green Home | 6% | EPC E–G 0.8, gas/oil heating 0.5, energy spend per person 0.6, mortgage 0.5, income 0.2 | eco 0.6, change 0.4, headroom 0.3 |
| GP03 Solar | 5% | electricity per person 0.7, detached/semi 0.5, mortgage 0.4, not urban 0.3, income 0.5 | eco 0.6, headroom 0.5 |
| GP04 Paperless | 25% | app logins 0.8, younger 0.4, age 65+ −0.3 | digital affinity 0.6, change 0.3 |

**Sanity check** (`make check-labels`, 10,000 customers, validation AUC on each eligible group). A quick untuned LightGBM beats the best single signal for EV (0.72 vs 0.68), Green Home (0.68 vs 0.58) and Solar (0.68 vs 0.67), and ties on Paperless (0.81 vs 0.82). Green Home has only 23 validation take-ups, because real EPC ratings make just 26% of customers eligible, so its numbers are noisy.

## Leakage controls

- Features use only data dated on or before the cutoff. Labels come only from the label window.
- Attributes are snapshots as of the cutoff (D3). Products opened after the cutoff are ignored.
- Transactions caused by take-up (EV charging, new loan repayments) only occur after the cutoff.
- Splits are customer-level. The test set is scored once.
- Hidden traits and `true_category` are never features.

## Assumptions and limitations

- **No public source** exists for these, so they're assumptions (`config/calibration_targets.yaml`):
    - paper statements: 25% of customers
    - marketing consent: 85%
    - car loans: 35% of car-owning households
- **Use-case flags are empty in v1,** so EV Finance is never shown to EV owners or to customers without a car.
- **Several sources are England-only and used as UK proxies:** age, tenure, property type, EPC, heating fuel, car availability and urban/rural.
- **"Urban" means London only** (the 2021 rural-urban classification has no conurbation class). Other big cities count as "town".
- **EPC is modelled for the whole stock** (English Housing Survey), and A and B are generated together as B.
- **The official sources disagree on mileage** (ONS spend implies about 5,300 miles per car; the NTS says 7,100). We match ONS spend.
- **Energy prices** are the day-weighted average of the four Ofgem cap quarters in the window, and one dual-fuel split ratio is used for everyone.
- **The merchant list is small** (about 45 templates), so FR2 categorisation scores are optimistic until it's enlarged.
- **Labels are synthetic.** Take-up rates (8% / 6% / 5% / 25% within 3 months) are deliberately generous so the model has enough positives, and real green-finance take-up is likely far lower.
