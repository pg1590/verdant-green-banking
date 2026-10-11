# Carbon methodology and current coverage

P2's factors live in `config/emission_factors.csv`, with source, release, units
and reference for each one. The engine is `src/carbon/engine.py`. It is pure
Python, with no ML and no LLM. Every estimate carries a method, assumptions
(including the baseline), a one-year horizon and the DESNZ 2026 `factor_edition`.
If an input is missing or the case is outside a documented method, the result is
`INSUFFICIENT_DATA` with a reason code. It never becomes zero.

All figures are **annual, first-year, use-phase potential** for one customer. They
are not guaranteed or realised savings, and they are not added up across
products for one customer (see Portfolio totals).

## Summary by product

| | EV Finance (GP01) | Green Home (GP02) | Solar (GP03) | Paperless (GP04) |
|---|---|---|---|---|
| Intervention | Replace the current petrol, diesel or hybrid car with an average BEV | Replace a gas boiler with an air-source heat pump | Install a reference 3.5 kWp south-facing PV system | Switch paper statements to digital |
| Baseline (from customer data) | Estimated km/year from road-fuel spend | Gas kWh/year from bills | Electricity kWh/year from bills, region | Paper statements in the 12-month window |
| Outputs | kg CO2e | kg CO2e, kWh | kg CO2e, kWh | sheets, kg CO2e |
| Key sources | DESNZ 2026 car factors | EoH heat pump trial; BEIS boiler IA; DESNZ 2026 | DESNZ Solar Roadmap; PVGIS; MCS MGD 003; DESNZ 2026 | DESNZ 2026 paper factor; team lower bound |

## EV Finance (GP01)

`kg CO2e/year = km/year x (ICE kg/km - BEV generation kg/km - BEV T&D kg/km)`

Mileage comes from road-fuel spend, average pump prices and fuel economy derived
from the DESNZ factors. EV charging is excluded. The estimate covers the use
phase only: no manufacturing, upstream fuel, rebound or financing.

Unavailable when: the main car is not petrol, diesel or hybrid
(`NO_DOCUMENTED_ICE_REPLACEMENT_BASELINE`), mileage is missing or zero, or the
household has more or fewer than one car
(`SINGLE_REPLACED_VEHICLE_MILEAGE_NOT_ESTABLISHED`). Household mileage cannot be
assigned to one replaced car.

## Green Home (GP02): gas boiler to air-source heat pump

EPC alone does not give a saving, so we model one documented intervention and
apply it to the customer's own gas use.

```
heat delivered (kWh)   = gas kWh x 0.84        boiler in-situ efficiency
heat pump electricity  = heat / 2.80            ASHP seasonal performance factor
kg CO2e avoided        = gas kWh x 0.18231 - heat pump kWh x (0.13096 + 0.01299)
energy (kWh)           = gas kWh - heat pump kWh   (less purchased energy)
```

| Input | Value | Source |
|---|---|---|
| Boiler in-situ efficiency | 84% | BEIS impact assessment BEIS067(C)-22-CHD, 13 Dec 2022, para 43 (EST 2009 field trial mean was 82.5% for combis) |
| ASHP SPF (SPFH4) | 2.80 median | Energy Systems Catapult, Electrification of Heat Demonstration, interim insights Mar 2023 (n=291, IQR 2.53-3.09) |
| Gas, grid, T&D factors | 0.18231, 0.13096, 0.01299 kg/kWh | DESNZ 2026 |

The assumptions list states the gas displaced and the electricity added
separately, because less purchased energy is not the same thing as avoided
emissions. All gas is treated as heating and hot water (cooking is not
separated). A dual-fuel apportioned baseline is flagged as less certain.

Excluded: refrigerant leakage, manufacturing, installation, fabric upgrades and
rebound. A real design needs a heat-loss survey.

Unavailable when: heating fuel is not gas (`NO_DOCUMENTED_GAS_HEATING_BASELINE`),
gas use is missing or zero (`MISSING_GAS_USE`), gas use is above twice Ofgem's high
benchmark of 17,000 kWh (`GAS_USE_OUTSIDE_PLAUSIBLE_RANGE`, D29), or the stated use case is a
retrofit without a documented saving, such as insulation
(`NO_DOCUMENTED_SAVING_FOR_THIS_RETROFIT`).

Worked example: 11,500 kWh gas gives 9,660 kWh heat and 3,450 kWh of heat pump
electricity. Avoided = 2,096.6 - 496.6 = **1,599.9 kg CO2e/year**; purchased
energy falls by **8,050 kWh/year**.

## Solar Finance (GP03): reference 3.5 kWp design

```
generation (kWh)     = 3.5 kWp x regional yield (kWh/kWp/year)
self-consumed (kWh)  = generation x MCS self-consumption share
kg CO2e avoided      = self-consumed kWh x (0.13096 + 0.01299)
```

| Input | Value | Source |
|---|---|---|
| System size | 3.5 kWp, south-facing | DESNZ UK Solar Roadmap, June 2025, footnote 26 ("a typical 3.5kW south-facing installation") |
| Yield | 853-1,042 kWh/kWp/year by region | PVGIS v5.3, one city per region, 14% loss, 35 degree pitch (`data/reference/pvgis/`) |
| Self-consumption | 14%-37% | MCS MGD 003 Issue 2.0, PV-only column of the "in half the day" tables 7-10 to 7-18 (`config/solar_self_consumption.csv`) |

The self-consumed share depends on both generation and the customer's own annual
electricity use, as MGD 003 describes. Occupancy is unknown, so we use the
"in half the day" tables, which MGD 003 says to use when occupancy is unknown.
**Only electricity used at home counts.** Exported electricity gets no credit,
because the grid-displacement effect of exports is not documented here.

Excluded: panel manufacturing, installation, degradation and batteries. Roof
suitability, shading and orientation are not surveyed. This is an illustrative
design, not a quote.

Unavailable when: the customer has or may have solar
(`EXISTING_OR_UNKNOWN_SOLAR_INSTALLATION`), the property is a flat or unknown
(`NO_DOCUMENTED_ROOF_FOR_REFERENCE_DESIGN`), the region or electricity use is
missing, or electricity use is outside the MGD 003 range of 1,500-5,999 kWh
(`OUTSIDE_MCS_SELF_CONSUMPTION_RANGE`).

Worked example (London, 3,200 kWh/year): generation 3.5 x 1,018.84 = 3,566 kWh.
MCS table 7-13, row 3,300-3,599 kWh gives 23%, so 820.2 kWh is used at home and
avoided emissions are **118.1 kg CO2e/year**.

## Paperless (GP04)

`kg CO2e/year = statements x sheets per statement x 0.005 kg x 1,343.59 kg/tonne / 1,000`

We found no published page count for a bank statement. We use **one sheet per
statement**, the minimum a paper statement can use, so the estimate is a lower
bound and scales linearly with the true count. The factor is DESNZ 2026 primary
paper production. Postage, printing, envelopes and digital-service emissions are
excluded (D16).

Unavailable when: the statement count is missing, or no paper statements were
sent in the window (`NO_PAPER_STATEMENTS_OBSERVED`). There is no zero estimate.

Worked example: 12 statements = 12 sheets = **0.081 kg CO2e/year**. The CO2e is
tiny, so the dashboard reports paper sheets as the main paperless metric.

## Shared physical conversions

P1's feature builder uses P2's predicted categories and metadata. The shared
conversion helper separates EV charging from road fuel, apportions dual-fuel
bills, subtracts standing charges and uses day-weighted Ofgem price-cap tariffs
from `config/energy_prices.csv`. These GB direct-debit averages are a proxy, not
each customer's tariff. The Jul-Sep 2026 tariff row is used only by trend
snapshots after the June cutoff; the canonical feature store uses Jul 2025 to
Jun 2026.

## Portfolio totals

P5 takes one highest-ranked eligible finance intervention per customer. If its
impact is missing, that customer has no coverage; another product is not
substituted. This also stops overlapping home interventions (heat pump and solar)
being double-counted. Paperless sheets and CO2e are reported separately as
digital impact. Each metric shows its own customer coverage. Totals are annual,
unweighted potential for the displayed cohort, not expected conversions or
achieved reductions.

## Why an estimate is unavailable

Every INSUFFICIENT_DATA result carries a reason code, and `UNAVAILABLE_REASONS` in
`src/carbon/engine.py` gives the plain-English sentence shown to customers and staff. P5 keeps the
code in the recommendation (`impact.reason`), so the UI can say why instead of only "no estimate".

## Tests

`tests/carbon/test_impacts.py` checks each formula against hand calculations
from the source tables, checks every missing or unsupported input, and checks
that every estimate passes P5's impact validation.
