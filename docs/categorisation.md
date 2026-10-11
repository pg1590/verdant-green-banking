# FR2 transaction categorisation

Every transaction gets **exactly one** of eight mutually exclusive categories. The machine-readable version of this page is `config/categories.yaml`; keep the two in sync. Results are in `reports/categorisation_eval.md`.

## Boundary decisions

| Category | Includes | Goes elsewhere |
|---|---|---|
| **Fuel** | Petrol, diesel and LPG at the pump; public and home **EV charging** | Fuel-station **shop purchases** → Other; car washes → Automotive |
| **Electricity** | Single-fuel electricity bills and top-ups | **Dual-fuel** bills → Utilities |
| **Gas** | Single-fuel mains gas bills and top-ups | Dual-fuel → Utilities; heating oil and LPG deliveries → Utilities |
| **Public Transport** | Rail, bus, coach, tram, underground, ferry, contactless transit (e.g. TfL) | **Taxis and ride-hailing** → Other (D17); flights → Other |
| **Home Improvement** | DIY and building materials, tradespeople, energy installers | Furniture and appliances → Other |
| **Automotive** | Servicing, MOT, repairs, parts, tyres, car washes, car insurance, parking, tolls, vehicle tax | Fuel and charging → Fuel; vehicle loan repayments → Other |
| **Utilities** | Dual-fuel energy, water, broadband, phone, council tax, heating oil and LPG | Single-fuel electricity or gas |
| **Other** | Everything else, including taxis, flights, streaming, furniture, loan repayments, HMRC tax | none |

Why these choices:
- **Fuel includes EV charging.** Both are paid for road energy, so a customer switching to an EV moves spend *within* Fuel. That keeps the EV signal and the carbon baseline consistent. Because P1's `VEHICLE_FUEL_CONFLICT` must come from **road** fuel only, every Fuel row also carries `is_ev_charging`.
- **Fuel-station shop purchases go to Other.** "SHELL 4421 LEEDS" for £3.80 is a sandwich, not fuel. They are separated by the description ("SHOP PURCHASE"), the convenience-store MCC 5499 and, in the model, the amount. Counting them as Fuel would inflate fuel spend for non-drivers.
- **Dual-fuel bills go to Utilities (D6).** The category has to be single, and a dual-fuel bill is not a single-fuel bill. The electricity and gas *signal* isn't lost: those rows carry `is_dual_fuel`, and the feature layer splits them into estimated kWh using the Ofgem typical-consumption ratio and flags the estimate.
- **Energy suppliers' names are not fuel hints.** British Gas, EDF, Octopus and others sell electricity, gas and dual fuel, so "BRITISH GAS" alone does not mean Gas. The fuel type comes from the description ("DD ELECTRICITY", "GAS A/C", "DUAL FUEL") or from the merchant name once the supplier's name is removed. A British Gas boiler installation (MCC 1711) is Home Improvement, not a bill.
- **MCC 4900 covers all utilities**, and MCC 4899 covers broadband *and* streaming, so for those codes the text decides.

## Method

### Rules baseline (`src/categorisation/rules.py`)
1. **Clean the text:** upper case; drop apostrophes, punctuation, reference numbers and store numbers; join "B & Q" into "B&Q"; drop legal and payment noise words from merchant names (LTD, PLC, UK, DD, CONTACTLESS…). "SHELL 4421 LEEDS" → "SHELL LEEDS".
2. **Apply ordered rules**, first match wins:

   | Step | Rule | Strong? |
   |---|---|---|
   | 1 | Forecourt shop purchase (description) → Other | yes |
   | 2 | EV charging (MCC 5552 or charging network name) → Fuel, `is_ev_charging` | yes |
   | 3 | Energy supplier bill + fuel hint → Electricity / Gas / Utilities (`is_dual_fuel`) | yes |
   | 4 | Unambiguous MCC → its category | yes |
   | 5 | Merchant or description keywords; the longest matching phrase wins | no |
   | 6 | Ambiguous MCC (4899, 4900, 4816, 6300, 9311, 9399) | no |
   | 7 | Anything else → Other | no |

Rules run once per distinct (merchant, description, MCC), so millions of rows are cheap.

### Model (`src/categorisation/model.py`)
- **Inputs:** TF-IDF of character 2–5-grams of the cleaned merchant name and description, one-hot MCC (with an explicit "missing" value) and channel, and log(amount). `true_category` is never an input.
- **Classifier:** logistic regression (C = 4, class-balanced weights, seeded). Identical inputs are collapsed into weighted rows before training, so training cost depends on the number of distinct merchant strings, not transactions.
- **Final answer:** any rule that recognises the transaction (steps 1–5) decides. The model only replaces a rule fallback (an ambiguous MCC or the default "Other", steps 6–7), and only when its probability is at least 0.9 (`MODEL_MIN_CONFIDENCE`). It therefore handles merchants the rules don't know.
- **Why this order (model `fr2_tfidf_lr_v2`):** in v1 the model overrode everything except the strong rules. On the full 10,000-customer run that made the final categoriser worse than the rules alone: unseen-brand macro-F1 81.6% vs 100%, missing-MCC accuracy 75.6% vs 100%, hand-labelled accuracy 93.3% vs 99.0%. The main errors were Other → Fuel (6,994 transactions) and Automotive → Utilities (6,277). The slides allow rules, ML or both; the evaluation justifies rules first.
- **Training population:** train-split customers only. Validation and test customers' spend features therefore come from a categoriser that never saw their labels.

### Output (`data/processed/transactions_categorised.parquet`)
The input transactions, minus `true_category`, plus:

| Column | Meaning |
|---|---|
| `category` | One of the eight categories |
| `category_confidence` | 1.0 when a strong rule (steps 1–4) decided; otherwise the model's probability of the chosen category |
| `category_source` | `rule:<RULE>` naming the rule that decided, or `model` |
| `is_ev_charging` | Fuel row that is EV charging. P1: road fuel = Fuel and not this |
| `is_dual_fuel` | Utilities row that is a dual-fuel energy bill. P1: apportion these (D6) |

Run it with `python -m src.categorisation.model` (stage 2 of `scripts/run_pipeline.py`). It reads `data/raw/` if P1's generated data is there, otherwise the mock sample in `data/sample/`.

## Evaluation (goes into report G)

`python -m src.categorisation.evaluate` writes `reports/categorisation_eval.md` and `.json`. Use `--split test` once, at the end.

- **Split by merchant brand (D4).** About 25% of the merchant brands in each category (at least one is always kept) are removed from training. The validation customers' transactions are scored separately for brands **seen** and **never seen** in training. The seen-brand figures are what a transaction-level split would show; the unseen-brand figures are the honest ones.
- **Three systems compared:** rules baseline, model only, and the final categoriser.
- **Metrics:** macro-F1, accuracy, per-class F1 and a confusion matrix.
- **Boundary cases** reported separately: missing MCC, EV charging, fuel-station shop purchases, energy-supplier bills and dual-fuel bills.
- **Independent check:** a 105-row hand-labelled set (`tests/categorisation/fixtures/hand_labelled_transactions.csv`), written without reference to P1's generator. It covers messy variants and traps (BP Pulse charging, "SCOTTISHPOWER" gas account, a British Gas boiler installation, Netflix on the broadband MCC, a refund). Every error is listed in the report.
- **Labels:** `true_category` is used for training and evaluation only, never as a feature.

## Limitations

- **Synthetic text.** Merchant strings come from a generator and are cleaner and less varied than real bank data. Scores on unseen brands are the best guide, but real-world accuracy would be lower.
- **The keyword lists use real UK brand names**, and so may P1's generator. Overlap makes the rules look better than they would on genuinely new merchants. The model-only column and the hand-labelled set show how much rests on that knowledge.
- **The boundaries are our own choices**, not a bank or regulatory standard. Another team could reasonably put EV charging in Electricity or car insurance in Other.
- **Energy fuel type depends on hints in the text.** A supplier bill with no "ELEC", "GAS" or "DUAL FUEL" hint cannot be split reliably. The rules fall back to Utilities and the model guesses from similar bills.
- **Merchant brand = first word of the cleaned name.** This can merge unrelated merchants that share a first word, or split one brand that is written in several ways.
- **`category_confidence` is a model probability, not a calibrated guarantee**, and strong-rule rows report 1.0 by convention.
- **The rules alone score 100% on P1's generated data** because the generator's merchants are brands in the keyword lists. The model's real contribution can only be judged on merchants outside those lists; the model-only column shows how much rests on brand knowledge.
- **The hand-labelled set is no longer untouched.** It was used to diagnose the v1 selector problem, so its v2 score is not an independent test result.
