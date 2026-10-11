# Integration and performance check · 11 October 2026

State checked: `main` with P3's v3 models, the new web app, and the fixes from the stress test
(listed below). Seed 42, offline explanations unless stated. Rerun the numbers at the freeze with
the commands below.

## Stress-test findings and fixes

| Finding | Before | After |
|---|---|---|
| Runs below about 1,290 customers crashed (scenario customers have no split) | `--n 1000` failed P3's 99% check | Scenario customers are excluded from the check; `--n 300` completes in 25 s, 0.44 GB |
| Memory use | 6.5 GB peak at 10k | 3.75 GB peak for the full pipeline: the feature build reads only the 4 columns it needs from the categorised file and no longer copies text columns |
| Portfolio endpoints queued under load | about 1 s each; slowest of 50 concurrent clients 35 s | cached until feedback or the date changes: 1-9 ms; 150 concurrent portfolio requests, slowest 0.54 s, no errors |
| Unavailable impact lost its reason in the API | reason dropped | reason code kept and explained in plain English in the UI |
| Implausible gas estimates (73 customers above 34,000 kWh) | used for heat-pump estimates (2.8% of Green Home potential) | not used: `GAS_USE_OUTSIDE_PLAUSIBLE_RANGE` (D29) |
| 3,585 NumPy deprecation warnings in the tests | `pd.Timedelta(days=...)` with NumPy 2.5 | 0 warnings (`pd.Timedelta(n, unit='D')`) |

The categoriser finding (rules beat the rules + ML mix) is being fixed by P2 separately.

## Pipeline

`python scripts/run_pipeline.py --fast` on a Windows laptop: about 5 minutes on an idle machine,
peak memory 3.75 GB. Stages: generate 10,000 + 13 scenario customers (4.87M transactions),
train and evaluate the categoriser, feature store and Customer 360, train and score four models,
impacts, four dated trend snapshots, bundle export and offline explanations. The shipped v3 models
were trained by P3 with the full grid; their manifests match the local feature file, so
`python -m src.models.predict` scores with them without retraining.

## Results on the full population (v3 models)

- **Opportunities:** 6,934 customers have at least one shown opportunity, 11,935 in total
  (EV 4,968; Green Home 2,210; Solar 3,366; Paperless 1,391).
- **Impact:** estimates exist for 3,776 (EV), 8,010 (Home), 5,094 (Solar) and 2,515 (Paperless)
  customers before display rules. Counting one top finance opportunity per customer, 3,807 of 6,525
  have a CO2e estimate: about 4,619 t CO2e a year of potential, plus 9.1 GWh of energy (1,748
  customers). Paperless: 16,692 sheets a year (1,391 customers).
- **Optional products** (rule-based signals, no propensity): Green Investments 1,159 customers,
  Green Loans 1,822 (83 home-charger and 1,758 efficient-appliance use cases).
- **Trend:** 6,934 / 6,907 / 6,902 / 6,906 customers with an opportunity at the June to September
  month-ends; the June point equals the dashboard summary.

## Tests

- 303 automated tests pass, with no warnings (`python -m pytest -q tests`, about 2 minutes).
- Scenario customers C90001-C90013 are checked against their expected show/suppress outcome.
- Impact formulas are checked against hand calculations; every unavailable reason has customer text.
- Counts reconcile across summary, opportunity table and API; the cache refreshes after feedback,
  including feedback written by another process.
- What-if, optional products, ask-the-portfolio (with invented-number rejection) and the
  engagement funnel each have tests, including the API routes.

## Performance

`python scripts/benchmark.py` on the full bundle (10,013 customers, 47 MB): load and validate 1.7 s;
`/customers/{id}/insight` 2 ms; `/portfolio/summary` 1 ms, `/portfolio/opportunities` 1 ms,
`/segments` 9 ms after the startup warm-up. Under 50 concurrent clients per portfolio endpoint the
slowest request took 0.54 s with no errors; per-customer insight requests ran at about 180 per
second. Per-customer work is constant time because scores, eligibility and impact are computed in
batch. The next step for much larger portfolios is an indexed store (SQLite or PostgreSQL) instead
of one JSON bundle.

## Known limitations

- 44% of eligible Paperless rows have no reason that passes its evidence gate and are hidden.
- The held-out test set was scored for v2 and then once for v3 (logged); it is not untouched.
- Optional products have no take-up model and no impact estimate, by design (D26).
- All data and labels are synthetic; see `docs/responsible_ai.md`.
