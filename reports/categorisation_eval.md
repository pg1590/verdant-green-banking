# FR2 categorisation evaluation

Generated 2026-10-10 by `python -m src.categorisation.evaluate --split val` on `data/raw`. Model `fr2_tfidf_lr_v1`.

Trained on 2,240,940 transactions of train-split customers, excluding 14 held-out brands (BIRMINGHAM, BP, BRISTOL, CAR, EDF, LIVERPOOL, LONDON, OVO, PRET, SHELL, STAGECOACH, TESCO, VIRGIN, WICKES). Scored on 736,485 transactions of `val` customers.

## Headline: macro-F1 / accuracy

| Transactions | n | Rules baseline | Model only | Final (rules + model) |
|---|---|---|---|---|
| all | 736,485 | 100.0% / 100.0% | 92.7% / 95.7% | 96.7% / 98.1% |
| seen brands | 485,238 | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| unseen brands | 251,247 | 100.0% / 100.0% | 65.5% / 87.3% | 81.6% / 94.4% |

Seen brands approximate a transaction-level split; unseen brands are the honest test (D4).

## Per-class F1 (all transactions)

| Category | Support | Rules | Model | Final |
|---|---|---|---|---|
| Fuel | 61,805 | 100.0% | 85.8% | 94.2% |
| Electricity | 13,076 | 100.0% | 99.5% | 100.0% |
| Gas | 9,300 | 100.0% | 100.0% | 100.0% |
| Public Transport | 91,620 | 100.0% | 99.7% | 99.9% |
| Home Improvement | 9,106 | 100.0% | 85.5% | 98.9% |
| Automotive | 25,501 | 100.0% | 77.7% | 85.7% |
| Utilities | 77,603 | 100.0% | 96.0% | 96.1% |
| Other | 448,474 | 100.0% | 97.4% | 99.2% |

## Boundary cases (accuracy)

| Slice | n | Rules | Model | Final |
|---|---|---|---|---|
| missing MCC | 57,507 | 100.0% | 74.5% | 75.6% |
| EV charging | 3,161 | 100.0% | 100.0% | 100.0% |
| fuel-station shop purchases | 13,079 | 100.0% | 0.0% | 100.0% |
| energy-supplier bills | 32,003 | 100.0% | 99.6% | 100.0% |
| dual-fuel bills | 9,627 | 100.0% | 100.0% | 100.0% |

## Confusion matrix, final categoriser (all)

Rows are the true category, columns the predicted one.

| true \ pred | Fuel | Electricity | Gas | Public Transport | Home Improvement | Automotive | Utilities | Other |
|---|---|---|---|---|---|---|---|---|
| **Fuel** | 61441 | · | · | · | 169 | 34 | · | 161 |
| **Electricity** | · | 13076 | · | · | · | · | · | · |
| **Gas** | · | · | 9300 | · | · | · | · | · |
| **Public Transport** | 266 | · | · | 91354 | · | · | · | · |
| **Home Improvement** | 1 | · | · | · | 9084 | 15 | 6 | · |
| **Automotive** | 1 | · | 2 | · | · | 19221 | 6277 | · |
| **Utilities** | · | · | · | · | · | · | 77603 | · |
| **Other** | 6994 | · | · | 1 | 3 | 90 | · | 441386 |

## Confusion matrix, final categoriser (unseen brands)

Rows are the true category, columns the predicted one.

| true \ pred | Fuel | Electricity | Gas | Public Transport | Home Improvement | Automotive | Utilities | Other |
|---|---|---|---|---|---|---|---|---|
| **Fuel** | 29083 | · | · | · | 167 | 34 | · | 161 |
| **Electricity** | · | 4473 | · | · | · | · | · | · |
| **Gas** | · | · | · | · | · | · | · | · |
| **Public Transport** | 266 | · | · | 22646 | · | · | · | · |
| **Home Improvement** | 1 | · | · | · | 2226 | 15 | 6 | · |
| **Automotive** | · | · | 2 | · | · | · | 6277 | · |
| **Utilities** | · | · | · | · | · | · | 27643 | · |
| **Other** | 6994 | · | · | 1 | · | 87 | · | 151165 |

Who decided the final category: rule (strong): 67.4%, model: 32.6%.

## Independent hand-labelled set

105 transactions written by hand, independently of P1's generator, covering the boundary cases (`tests/categorisation/fixtures/hand_labelled_transactions.csv`).

| | Rules | Model | Final |
|---|---|---|---|
| macro-F1 | 99.3% | 61.0% | 93.3% |
| accuracy | 99.0% | 69.5% | 93.3% |

Errors of the final categoriser:

| txn | merchant | description | MCC | true | predicted | decided by | note |
|---|---|---|---|---|---|---|---|
| S0011 | BLACK HORSE LTD | DIRECT DEBIT |  | Other | Utilities | model | Car finance repayment (team decision: Other - a loan repayment is not spend) |
| S0013 | DVLA-AB12CDE | DIRECT DEBIT | 9399 | Automotive | Utilities | model | Vehicle tax; contains a number plate |
| S0022 | ADMIRAL INSURANCE | DIRECT DEBIT MOTOR POLICY | 6300 | Automotive | Utilities | model | Car insurance |
| S0023 | DART CHARGE | ONLINE PAYMENT | 4784 | Automotive | Public Transport | model | Road toll |
| S0050 | R JONES PLUMBING HEATING | FASTER PAYMENT |  | Home Improvement | Utilities | model | Sole trader paid by bank transfer; no MCC and no known brand |
| S0058 | TV LICENCE MBP | DIRECT DEBIT |  | Utilities | Gas | model | TV licence (not listed by team; labelled Utilities) |
| S0081 | AA INSURANCE | DIRECT DEBIT HOME POLICY | 6300 | Other | Utilities | model | Motoring brand but a home insurance policy |
