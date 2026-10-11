# FR3 Propensity Models: Evaluation Report

> Owner: P3. Metric tables are auto-generated into `reports/ml_eval_tables.md` by
> `python -m src.models.train`; test-set numbers by `python -m src.models.run_test` (run once).
> Numbers below: models `v3`, trained on P1's seeded 10,000-customer dataset (`make pipeline`).

## 1. Task
For each customer and each required product (GP01 EV Finance, GP02 Green Home, GP03 Solar
Finance, GP04 Paperless) output a calibrated **propensity** (probability of take-up in the
label window), a **confidence** and **reason codes**. Propensity never decides eligibility.

## 2. Data, splits and leakage control
- Source: P1's synthetic generator (`docs/data.md`). Features up to the 2026-06-30 cutoff,
  labels from the 2026-07-01 to 2026-09-30 window.
- Splits by **customer** 70/15/15 (P1); `src/models/data.py` asserts each customer is in one
  split, and that features, labels and splits describe the same customers (≥ 99% overlap,
  no duplicate IDs, 0/1 labels). A stale or mismatched path fails instead of silently
  inner-joining to a few rows.
- **Never model inputs**, enforced when the config loads (`check_feature_policy`): age (D9),
  IDs, dates, flags, use-case fields, labels, `true_category`, persona and hidden traits
  (D1), including any encoding derived from them.
- **Model population:** P5's catalogue eligibility (`purpose="training"`). Each model is
  trained and evaluated only on customers for whom the product is plausible (e.g. homeowners
  for Solar). Including structurally-zero customers would inflate AUC by learning the
  eligibility rule instead of propensity.
- Every manifest records the sha1 of the features, labels, splits and the four configs it
  was trained from (`input_sha1`); inference refuses models whose features changed.

## 3. Method
- **Two candidate families per product**, both tuned by 5-fold stratified CV on TRAIN only
  (scored by average precision):
  - LightGBM with **monotone constraints** on the economic drivers (e.g. fuel spend ↑ ⇒ EV
    propensity cannot ↓), small trees (2–8 leaves, ≥ 50 rows per leaf, L2) because each
    product has only ~100–450 train positives;
  - logistic regression on log-compressed inputs (median impute, signed log1p, standardise, L2).
- **Selection:** the family with the higher train-CV score ships, per product. VAL and TEST
  never choose the model. Logistic labels (P1's take-up rule is a logit) favour the linear
  model where signal is thin; LightGBM wins where interactions or thresholds matter.
- **Calibration:** `CalibratedClassifierCV(cv=5, ensemble=False)` on TRAIN. Platt (sigmoid)
  below 500 train positives, isotonic otherwise. The calibrator is fitted on out-of-fold
  predictions, then **one** model is refitted on all of TRAIN. That model is the one SHAP
  explains, so reason codes describe the model that scores. No class reweighting (it would
  distort probabilities).
- **Validation set** is used only for reporting and comparison. **Test set evaluated once**
  per frozen version.

### Baselines
- **B1 single-signal rule:** "signal > train p75" (fuel spend / gas kWh / electricity kWh / app logins).
- **B2 logistic regression** (median impute, standardise, L2; no log transform).

## 4. Confidence (documented method)
```
confidence = data_completeness × agreement × data_quality
agreement    = 1 − min(1, σ_logit / 2.0)       σ_logit: std of the 5 bootstrap models' predictions in log-odds
data_quality = 0.5 if a product-relevant conflict flag is set
               × 0.7  for GP01 if RECENT_CIRCUMSTANCE_CHANGE
               × 0.85 for GP02/GP03 if ENERGY_APPORTIONED_FROM_DUAL_FUEL
```
The spread is measured in log-odds because in probability units it grows with the score
itself, which would give the highest-propensity customers the lowest confidence. Missing
data, unstable estimates, contradictory data and estimated inputs each lower confidence;
P5 applies no second penalty. `score()` refuses input without completeness, conflict or
data flags rather than assume clean data.

## 5. Reason codes
Top positive SHAP contributors in log-odds (TreeSHAP for LightGBM, exact linear SHAP for
the logistic model) → fixed dictionary `config/reason_codes.yaml` → kept only if the raw value
passes the code's evidence gate (no `HIGH_FUEL_SPEND` for £20/month). Percentile gates are fitted
on each product's TRAIN cohort and frozen in the bundle (`reports/reason_code_thresholds.json`).
At most one code per feature (the first passing one, most specific first) and at most 3.

## 6. Results (validation, n = 15% of customers, model population only)

| Product | Shipped | Train-CV AP (lgbm / logreg) | ROC-AUC sel. | B2 logreg | B1 rule | PR-AUC sel. | ECE | Reason coverage |
|---|---|---|---|---|---|---|---|---|
| GP01 | logreg | 0.205 / **0.216** | 0.741 | 0.746 | 0.624 | 0.242 | 0.010 | 85.9% |
| GP02 | lgbm | **0.146** / 0.140 | 0.673 | 0.790 | 0.472 | 0.094 | 0.007 | 97.7% |
| GP03 | logreg | 0.122 / **0.130** | 0.716 | 0.723 | 0.520 | 0.200 | 0.012 | 88.4% |
| GP04 | lgbm | **0.527** / 0.522 | 0.791 | 0.795 | 0.692 | 0.565 | 0.042 | 55.9% |

Full tables with 95% bootstrap CIs, Brier, precision and lift: `reports/ml_eval_tables.md`.
Validation positives are few (GP02 19, GP03 37), so the CIs are wide (GP02 B2 vs selected
overlap). The shipped model beats the single-signal rule everywhere, and its train-CV score
is at least that of the other family by construction. Against B2 on VAL it is within the CI.

## 7. Driver recovery
Top-5 mean |SHAP| on VAL against P1's label rules (`config/label_rules.yaml`):

| Product | Top SHAP features | P1's label drivers | Reading |
|---|---|---|---|
| GP01 | income, vehicle loan, prior fuel spend, urban, fuel spend | fuel per vehicle 0.6, vehicle loan 0.6, 2+ cars 0.4, fuel 0.3, urban 0.3, income 0.3 | recovered, but income is over-weighted (it also proxies hidden financial headroom) |
| GP02 | home-improvement spend, energy per person, income, electricity kWh, mortgage | EPC E–G 0.8, energy per person 0.6, gas/oil 0.5, mortgage 0.5, income 0.2 | partly: EPC and heating fuel are under-used; home-improvement spend is not a label driver (correlated with persona) |
| GP03 | income, electricity per person, mortgage, detached/semi, energy per person | electricity per person 0.7, detached/semi 0.5, income 0.5, mortgage 0.4, not urban 0.3 | recovered |
| GP04 | app logins, web sessions, income | app logins 0.8, younger 0.4 (age is excluded, D9) | recovered |

Hidden traits (eco-mindedness etc.) carry a large share of the label signal and are deliberately
unobservable, which caps achievable AUC around 0.75–0.8.

The logistic models do not enforce the monotone directions. `sign_violations` in each manifest
lists features whose coefficient contradicts model.yaml: GP01 `num_vehicles` and
`spend_automotive_m` (collinear with mileage and fuel spend), GP03 `has_savings`. Their reason
codes only fire on high values, where a negative coefficient gives a negative contribution, so
none of them is ever emitted (see `reason_code_counts` in `metrics_val.json`).

## 8. Fairness slices
`reports/fairness_val.csv`: per region, age band, income band and urban/rural, the size,
positives, mean propensity, take-up, `calibration_ratio` (mean propensity ÷ take-up) and AUC
with a bootstrap 95% CI. AUC is reported **only** for groups with ≥ 10 positives and ≥ 10
negatives (`auc_reliable`); below that it is blank rather than quotable noise. Most GP02/GP03
cells are below that line on a 15% validation split. Visible gaps worth monitoring:
GP03 is under-predicted for 60+ (calibration ratio 0.66) and Q5 (0.68); GP04 for 46–60 (0.65).
See `docs/responsible_ai.md`.

## 9. Test set (single run of v3, `outputs/test_predictions.csv`)

| Product | n | positives | ROC-AUC | PR-AUC | Brier | ECE | Lift@10% |
|---|---|---|---|---|---|---|---|
| GP01 | 1013 | 85 | 0.740 | 0.227 | 0.072 | 0.013 | 3.42 |
| GP02 | 387 | 25 | 0.729 | 0.215 | 0.058 | 0.010 | 2.38 |
| GP03 | 649 | 35 | 0.803 | 0.233 | 0.048 | 0.013 | 4.28 |
| GP04 | 343 | 61 | 0.812 | 0.515 | 0.122 | 0.070 | 3.31 |

The test split was first scored by the integration run on the earlier `v2` models (fixed
parameters, LightGBM only; ROC-AUC 0.710 / 0.746 / 0.759 / 0.811). v3 was then frozen without
reference to those numbers and evaluated once with `--force`, which is logged in
`outputs/test_runs.log`. The 6,000 rows follow the agreed eight-column schema.

## 10. Limitations
- Synthetic labels: metrics reflect the generator's assumptions; not comparable to other datasets.
- Propensity ≠ persuadability (no uplift modelling); no time-based backtest; no drift monitoring.
- Small positive counts make VAL comparisons noisy; the model choice therefore uses train CV.
- The confidence normaliser (2.0 log-odds) and data-flag multipliers are documented heuristics,
  not fitted; confidence has not been validated against observed error rates.
- GP04 has empty reasons for ~45% of eligible customers (paper statements push propensity
  down, so they are rarely a positive driver); P5 suppresses rows without reasons.
