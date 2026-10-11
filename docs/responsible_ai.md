# Responsible AI note

This note covers privacy, bias, explainability, hallucination, human
oversight and data limitations. Everything here describes a **synthetic-data
prototype**. Nothing below is a claim about a production system.

## Who decides what

| Decision | Made by | Can the LLM or memory change it? |
|---|---|---|
| Propensity (chance of take-up in 3 months) | ML models (P3) | No |
| Whether a product may be shown | Catalogue rules in `config/products.yaml` (P5) | No |
| Impact (kg CO2e, kWh, paper sheets) | Deterministic engine with sourced factors (P2) | No |
| Wording of the explanation | LLM, validated, with a template fallback (P4) | n/a |
| Order shown after feedback | Customer memory (P4, optional) | Only hides, cools off or demotes rows that are already eligible |

A propensity score never makes a product eligible. Suppressed products get no
customer-facing text at all.

## Privacy

- All customers, transactions and labels are synthetic. No real personal or
  banking data is used. Public sources are aggregate statistics (`DATA_SOURCES.md`).
- **The LLM sees very little.** Its input is built from one final recommendation:
  product, reason wording, formatted evidence values, impact display, certainty band
  and data date. It never receives the customer ID, raw scores, other products or
  raw transactions (`src/llm/payload.py`, `docs/llm_design.md`).
- **Free text never reaches the LLM.** Any evidence value that is not a number or a
  short category is withheld. The validator also rejects output that contains a
  customer identifier.
- **Provider.** The prototype uses the Gemini API free tier, where prompts may be used
  to improve Google's products. That is acceptable here only because payloads are
  synthetic and minimised. A real deployment would need an enterprise tier with no
  training on prompts, a data-processing agreement and UK/EU data residency.
- **Retention and erasure.** The committed LLM cache holds only minimised payloads
  and outputs. Customer feedback memory is a local SQLite file (git-ignored).
  `DELETE /customers/{id}/memory` or "Clear my feedback history" erases it.
- Hidden persona traits, labels and `true_category` are rejected if they appear in
  a customer bundle (`src/recommend/validation.py`).

## Bias and fairness

**Inputs.** Age is not a model input (D9); it is kept only to run these checks and
for the adult eligibility rule. Income band *is* a model input. In the synthetic
data, income is one of the drivers of take-up, so the models learn it. Income
band is a signal of interest, not an affordability check or credit approval.
Marketing consent and eligibility rules change who is shown a product, separately
from the model.

**What we measured.** `reports/fairness_val.csv` (P3, v3 models) gives, for each region,
age band, income band and urban/rural group: size, positives, mean propensity, observed
take-up, a calibration ratio (mean propensity / take-up) and AUC where there are enough
positives. It uses the validation cohort that is eligible for each product. We compare each
group's mean propensity with the highest group's, and flag ratios below 0.8 (the
four-fifths rule of thumb).

| Product | Largest gap | Ratio to top group | Group size | Observed take-up, low vs top group |
|---|---|---|---|---|
| EV Finance | Income Q1 vs Q5 | 0.21 | 177 | 3.4% vs 14.2% |
| Green Home | Income Q1 vs Q5 | 0.53 | 85 | 2.4% vs 11.7% |
| Solar | Income Q1 vs Q5 | 0.18 | 125 | 2.4% vs 11.1% |
| Paperless | Age 60+ vs 46-60 | 0.80 | 308 | 23.7% vs 46.8% |

(For Paperless the nominal top age group, 18-30, has one customer, so we compare with the
next group.) Several regional ratios fall below 0.8, but most regional groups have fewer
than 60 customers (Northern Ireland has 6-23), so those gaps are within noise.

**How to read this.** The income gaps follow the observed synthetic take-up, which was
generated with income as a driver: Q1 take-up is about a quarter of Q5 for EV and Solar.
Calibration by income band is mostly 0.8-1.0, so the models reproduce the labels rather
than add a bias. Some groups are under-predicted: Solar for 60+ (calibration ratio 0.66)
and Q5 (0.68), Paperless for 46-60 (0.65). That is still a real risk. A bank that used
these scores for outreach would show finance offers to lower-income customers less often.
We do not claim the models are fair. The ratios are diagnostic evidence, and real-world
fairness cannot be shown on synthetic labels.

**Controls in the prototype.**
- Scores are shown to relationship managers as review prompts, never as decisions.
- Eligibility rules are the same for every income band. The model only orders
  opportunities.
- Finance products always carry the notice that display is not credit approval.

**What a real deployment would add:** fairness checks on real outcomes, a review of
whether income should stay a model input, and monitoring of who is shown offers
over time.

## Explainability

Every shown recommendation carries:
- up to 3 reason codes from a fixed dictionary (`config/reason_codes.yaml`). A code
  appears only if the feature is among the top positive SHAP contributors **and**
  the raw value passes the code's evidence threshold.
- the evidence values behind each reason, the confidence, the data date, the model
  version and the policy version
- for impact: the method, every assumption (baseline included), the factor edition
  and the horizon (see `docs/carbon_methodology.md`)

The UI shows all of this under "Evidence and model provenance", with a Customer 360
panel and a suppression audit that lists why each hidden product was hidden.

Limits we state openly:
- Reason contributions explain the fitted model that scores (TreeSHAP for LightGBM,
  exact linear contributions for logistic regression) before its monotone calibration
  step. They rank the drivers but do not add up to the calibrated probability.
- Confidence is a documented heuristic (data completeness, ensemble spread and
  conflicts), not a statistical probability that the prediction is right.
- Some eligible rows have no reason that passes its evidence gate (validation coverage:
  EV 86%, Home 98%, Solar 88%, Paperless 56%). P5 suppresses those rows instead of showing an
  unexplained offer. This lowers reach, but the system never shows an offer it
  cannot explain.

Each explanation also records its provenance: LLM or template, model and prompt
version, validation failures on each attempt, withheld evidence and a timestamp.

## Hallucination controls

1. The LLM only explains. The prompt forbids predicting, calculating new numbers or
   implying eligibility.
2. Structured JSON output: `headline`, `text`, `reasons_used`.
3. A validator checks every output before anyone sees it (`src/llm/validator.py`):
   - every number must appear in the input
   - no invented reason codes, and every reason must be covered
   - impact must be labelled an estimate, and no figure may appear when impact is
     unavailable
   - banned language: approval or eligibility claims, guarantees, model internals,
     pressure selling
4. One retry with the validator's feedback, then a deterministic template that is
   tested to always pass.
5. Prompt injection: free-text fields never reach the LLM, and injection cases are
   in the evaluation set.

**Evidence.** On the earlier 48-case test split (before integration), Gemini passed
every hard check on the first attempt, with no injection leaks and no unsupported
claims according to the judge. In one development run Gemini wrote "Based on our
model…", which the validator caught before it reached a customer. The impact and
reason contracts changed at integration, so those earlier scores are not proof
for the current payloads. We therefore reran Gemini on 40 cases built from the real
integrated rows: all 32 shown cards passed every hard check on the first attempt, with
no injection leaks, and two independent judge families (Gemma 4 and gpt-oss-120b)
preferred them to the template in 74% and 84% of pairs (`docs/llm_design.md`). The
judges are LLMs, so their scores are relative evidence, not ground truth.

## Human oversight

- Actions say "Explore …" or "Review …", never "Apply" or "You qualify". Buttons open
  a review step and change nothing. Every row carries
  `eligibility_scope = DISPLAY_ONLY_NOT_CREDIT_APPROVAL`, and the policy notice is
  shown in the API and UI. Credit decisions stay with the bank's normal process.
- Conflicting data (for example, an EV owner buying petrol) suppresses only the
  affected products, and the suppression audit shows the reason.
- Customers without marketing consent are not shown finance products (D18).
- Low-certainty explanations use tentative wording, and staff see the confidence
  and evidence behind every insight.
- Customer feedback ("dismiss", "not relevant") is respected through cool-off
  periods. Memory never changes scores, eligibility or impact, and it cannot show
  an ineligible product.

## Data limitations

**Data and labels (P1)**
- Labels come from our own generation rules (take-up 8% / 6% / 5% / 25% in 3
  months, deliberately generous). The models learn these assumptions, and metrics
  describe this synthetic task, not real customers.
- Several sources are England-only and used as UK proxies: age, tenure, property
  type, EPC, heating fuel, car availability and urban/rural.
- Some rates have no public source and are assumptions: 25% paper statements, 85%
  marketing consent, 35% of car households with a car loan.
- Profile attributes are a single snapshot at 2026-06-30. The trend view rescores
  later months with the current models, which is not a record of past deployments.

**Categorisation (P2)**
- Merchant strings are generated and cleaner than real bank data. The merchant
  list is small, so scores on never-seen brands (macro-F1 81.6%) are the honest
  guide, and real accuracy would be lower.

**Impact (P2)**
- Spend-based mileage and energy use are proxies. They can be wrong for business
  fuel cards, shared households or unusual tariffs. Tariffs are GB averages.
- Solar uses a reference 3.5 kWp design and Green Home uses a heat-pump scenario.
  Neither is a site survey. Paperless uses a lower bound of one sheet per statement.
- All estimates are first-year, use-phase potential. They exclude manufacturing,
  installation, rebound and postage, and they are not realised savings.

**Models (P3)**
- Validation sets are small for some products (Green Home has 19 positives), so
  confidence intervals are wide.
- The model family is chosen per product on training cross-validation only: logistic
  regression for EV and Solar, monotone LightGBM for Home and Paperless. On validation,
  Green Home's LightGBM (PR-AUC 0.094) scored below the logistic baseline (0.177), but with
  19 positives the intervals overlap. Neither validation nor test was used to choose. See
  `reports/ml_eval.md`.
- The held-out test set was first scored with the earlier v2 models and then once with v3
  (logged). It is no longer untouched, and we say so.
- Propensity is not persuadability. There is no uplift model, so expected take-ups
  are not extra conversions caused by an offer.

**Explanations (P4)**
- Explanations are tested in English only, on synthetic rows.
