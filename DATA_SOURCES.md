# Data Sources — Green Customer Insights Platform

**Verdant · synthetic customer data and documented environmental factors**

This document lists every external data source used to calibrate the synthetic dataset and the carbon engine. All customer-level data in this project is **fully synthetic**. The sources below are published, aggregate statistics; no real personal or banking data is used.

The portfolio edition includes runtime solar lookup tables in `data/reference/pvgis/` and a synthetic sample in `data/sample/`. Large third-party workbooks and publications are linked below rather than bundled; download them from their publishers if auditing the original calibration sources. Download paths below describe the original team archive. Every number used by the carbon engine is in `config/emission_factors.csv` with its source and reference; calibration totals are in `config/calibration_targets.yaml`.

---

## Summary

| # | Source | Publisher | Used for | Licence |
|---|---|---|---|---|
| 1 | GHG Conversion Factors 2026 | DESNZ (UK Gov) | Carbon engine (FR4) | Open Government Licence v3.0 |
| 2 | Family Spending in the UK, **FYE 2025** | ONS | Transaction amounts by category (FR2) | Open Government Licence v3.0 |
| 3 | National Travel Survey 2025 | DfT | Annual mileage, public transport use | Open Government Licence v3.0 |
| 4 | Vehicle Licensing Statistics, Q1 2026 | DfT / DVLA | Vehicle fuel-type shares, EV owners | Open Government Licence v3.0 |
| 5 | Typical Domestic Consumption Values | Ofgem | Electricity / gas use bands | Ofgem terms (public publication) |
| 6 | English Housing Survey 2024-25 (+ other household totals) | MHCLG, ONS, DEFRA, DESNZ | Age, tenure, property type, EPC band, heating fuel, solar share, region and household size | Open Government Licence v3.0 |
| 8 | Weekly road fuel prices | DESNZ (UK Gov) | £ → litres for fuel spend; mileage estimate | Open Government Licence v3.0 |
| 9 | Energy price cap (unit rates) | Ofgem | £ → kWh for electricity and gas; dual-fuel split | Ofgem terms (public publication) |
| 10 | PVGIS v5.3 | European Commission JRC | Solar yield per kWp by UK region | Free use with attribution |
| 11 | UK Solar Roadmap (June 2025) | DESNZ (UK Gov) | Reference 3.5 kWp solar design | Open Government Licence v3.0 |
| 12 | MGD 003 Solar PV Self-Consumption, Issue 2.0 | MCS | Share of solar generation used at home (small cited extract) | MCS copyright; extract cited, not redistributed in full |
| 13 | Electrification of Heat Demonstration, interim insights (2023) | Energy Systems Catapult (DESNZ-funded) | Heat pump efficiency (SPF 2.80) | Public report |
| 14 | Improving boiler standards and efficiency, impact assessment (2022) | BEIS (UK Gov) | Existing gas boiler efficiency (84%) | Open Government Licence v3.0 |

Considered but not used: EPC open data (needs a sign-in; the English Housing Survey tables were enough) and the Kaggle Sparkov transaction set (merchant strings are generated instead). The numbering below keeps gaps for them.

---

## 1. UK Government GHG Conversion Factors 2026

- **Publisher:** Department for Energy Security and Net Zero (DESNZ)
- **Published:** 11 June 2026
- **Page:** https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2026
- **Collection (all years):** https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting
- **Methodology paper:** https://assets.publishing.service.gov.uk/media/6a2940543b15d05a7ce3202e/2026-GHG-conversion-factors-methodology-report.pdf
- **Files used:** Flat file (for code), Full set (for reading), Methodology paper, Major changes document
- **Downloaded on:** 2026-10-10 → `data/reference/desnz_ghg_2026/`. **Use the flat file revised in July 2026** (`ghg-conversion-factors-2026-flat-format-revised.xlsx`).

**Contains:** Emissions factors with units (kg CO₂e per litre / kWh / km / tonne) for fuels, UK electricity, passenger cars by size and fuel type (incl. BEV and PHEV), UK electricity for EVs, rail and bus, materials (incl. paper) and waste.

**Used for:**
- EV Finance — petrol/diesel car factor vs BEV factor × annual km
- Green Home — natural gas factor × kWh saved
- Solar Finance — UK grid electricity factor × kWh generated
- Paperless — paper factor × estimated paper weight per year

**Note:** The 2026 UK electricity factor is substantially lower than 2025 due to grid decarbonisation and a methodology change. State the edition used in all impact outputs. **The project uses the 2026 edition throughout (docs/decisions.md D7).** Every factor goes into `config/emission_factors.csv` with its table reference.

**Postage:** we have not found a letter-delivery factor in this set. Paperless impact therefore covers paper production only, unless a documented postage factor is sourced and cited (D16).

---

## 2. ONS Family Spending in the UK (April 2024 to March 2025; FYE 2025)

- **Publisher:** Office for National Statistics (Living Costs and Food Survey)
- **Release page:** https://www.ons.gov.uk/releases/familyspendingintheukapril2023tomarch2024
- **Files used:**
  - Workbook 1 — detailed expenditure and trends
  - Workbook 2 — expenditure by income
  - Workbook 3 — expenditure by region
  - Workbook 4 — expenditure by household characteristic (incl. housing tenure)
- **Downloaded on:** 2026-10-10 → `data/reference/ons_family_spending_fye2025/`. **Newer edition used: FYE 2025 (April 2024 to March 2025)**, released after this list was first drafted.

**Contains:** Average weekly household spending (£) by COICOP category, broken down by income decile/quintile, region, age, tenure and urban/rural.

**Used for:** Generating realistic transaction totals per category in the synthetic `transactions` table, varying by customer profile.

| ONS category | FR2 category |
|---|---|
| Petrol, diesel and other oils | Fuel |
| Electricity | Electricity |
| Gas | Gas |
| Rail, bus and coach fares | Public Transport |
| Maintenance and repair of dwelling | Home Improvement |
| Vehicle purchase, parts, servicing | Automotive |
| Water supply and other household services | Utilities |

---

## 3. DfT National Travel Survey (NTS) 2025

- **Publisher:** Department for Transport
- **Main release:** https://www.gov.uk/government/statistics/national-travel-survey-2025/nts-2025-household-car-availability-and-trends-in-car-trips
- **Ad-hoc tables:** https://www.gov.uk/government/statistical-data-sets/ad-hoc-national-travel-survey-analysis
- **Tables used:**
  - NTS0303 — average car trips and miles per person
  - NTS0601 — car trips and miles by age and sex
  - NTSQ01030 — car driver distance by region and rural/urban classification
- **Downloaded on:** 2026-10-10 → `data/reference/dft_nts/`: NTS0303, NTS0601, **NTS0318** (mode by region; used instead of NTSQ01030, which is not on the current NTS pages) and NTS9902 (household car availability by region).

**Contains:** Trips and miles travelled per person per year by mode, age, sex, region and rural/urban area. Covers residents of England.

**Used for:**
- Annual mileage per synthetic customer (reason code `HIGH_ANNUAL_MILEAGE`)
- Deriving fuel spend (mileage ÷ fuel economy × pump price)
- EV impact baseline (km driven per year)
- Public transport usage levels

---

## 4. DfT / DVLA Vehicle Licensing Statistics (January to March 2026)

- **Publisher:** Department for Transport, from DVLA administrative data
- **Release:** https://www.gov.uk/government/statistics/vehicle-licensing-statistics-january-to-march-2026/vehicle-licensing-statistics-january-to-march-2026
- **Tables used:** VEH1103 (licensed vehicles and zero-emission vehicles by body type); VEH series tables by fuel type and region (linked from the release page)
- **Downloaded on:** 2026-10-10 → `data/reference/dft_veh/`: VEH1103, VEH1104, VEH0141, VEH0130 (the large VEH0105, VEH0142 and VEH9901 tables were skipped).

**Contains:** Counts of licensed vehicles in the UK by fuel type, body type and region, with quarterly history.

**Used for:**
- Share of synthetic customers owning petrol / diesel / hybrid / BEV cars
- Exclusion scenario: existing EV owners must not receive EV Finance (FR7)
- Regional EV adoption differences for the business dashboard (FR6)
- "Changed circumstances" scenario (vehicle switch mid-period)

---

## 5. Ofgem Typical Domestic Consumption Values (TDCVs)

- **Publisher:** Ofgem
- **Latest values (from 1 July 2026):** https://www.ofgem.gov.uk/sites/default/files/2026-06/Summary-of-changes-to-energy-price-cap-1-July-to-30-September-2026-revised-TDCV.pdf
- **Review / consultation page:** https://www.ofgem.gov.uk/consultation/review-typical-domestic-consumption-values
- **Downloaded on:** 2026-10-10 → `data/reference/ofgem/` (TDCV letter of 1 July 2026)

**Contains:** Low / medium / high annual household consumption for electricity (single-rate and multi-register, e.g. Economy 7) and gas. Medium values from 1 July 2026: 2,500 kWh electricity, 9,500 kWh gas, 3,400 kWh multi-register.

**Used for:**
- Calibrating synthetic electricity and gas spend
- Thresholds for the `HIGH_ELECTRICITY_SPEND` and `HIGH_GAS_SPEND` reason codes (Ofgem high band)
- Fallback consumption baseline when a customer's own data is missing
- **Dual-fuel apportioning (D6):** the medium electricity : gas ratio, multiplied by price-cap unit rates (§9), splits a dual-fuel bill into estimated electricity and gas use

---

## 6. English Housing Survey 2024-25 and other household totals

- **Publishers:** MHCLG (English Housing Survey), ONS (Labour Force Survey households, Families and households), DEFRA (Rural England digest), DESNZ (Energy Trends table 6.3)
- **Exact tables:** cited line by line in `config/calibration_targets.yaml` (for example EHS annex table 2.3 for EPC bands, annex table 2.4 for property type, FA1201 for age and tenure, DA6101 for heating fuel)

**Used for:** the shares of age band, tenure, property type, EPC band, main heating fuel, existing solar, region, household size and urban/rural among synthetic customers. These are England-only sources used as UK proxies; the limitation is stated in `docs/data.md` and `docs/responsible_ai.md`. EPC is the EHS modelled distribution for the whole stock, with A and B combined and generated as B.

---

## 8. DESNZ Weekly Road Fuel Prices

- **Publisher:** Department for Energy Security and Net Zero
- **Page:** https://www.gov.uk/government/statistics/weekly-road-fuel-prices
- **Files used:** the weekly CSV/XLSX of UK average pump prices for ULSP (petrol) and ULSD (diesel)
- **Downloaded on:** 2026-10-10 → `data/reference/desnz_fuel_prices/` (weekly CSV 2018–2026)

**Contains:** UK average retail prices (pence per litre) for petrol and diesel, weekly.

**Used for:**
- Converting synthetic fuel spend (£) into litres, then km via fuel economy, for `est_annual_km` and the EV baseline
- Calibrating synthetic fuel transaction amounts
- `petrol_price_gbp_per_litre` / `diesel_price_gbp_per_litre` in `config/emission_factors.csv`: the mean over the feature window (2025-07-01 to 2026-06-30)

---

## 9. Ofgem Energy Price Cap

- **Publisher:** Ofgem
- **Page:** https://www.ofgem.gov.uk/energy-price-cap
- **Files used:** the price-cap summary for the cap period covering 30 June 2026 (unit rates and standing charges, direct debit)
- **Downloaded on:** 2026-10-10 → `data/reference/ofgem/price_cap_unit_rates_direct_debit.csv` (current page: Jul–Sep and Oct–Dec 2026). The four quarters inside the feature window (Jul 2025 to Jun 2026) come from archived copies of Ofgem's "Energy price cap explained" page and are listed with their sources in `config/energy_prices.csv`, together with Jul–Sep 2026 for the trend snapshots.

**Contains:** Maximum unit rates (p/kWh) and standing charges for electricity and gas, by cap period.

**Used for:**
- Converting electricity and gas spend (£) into kWh (`est_electricity_kwh_year`, `est_gas_kwh_year`)
- The dual-fuel split (D6), together with the TDCVs in §5
- Calibrating synthetic energy bill amounts

---

## 10. Solar yield: PVGIS (EU JRC)

- **Tool:** https://re.jrc.ec.europa.eu/pvg_tools/en/ (PVGIS v5.3 PVcalc API, SARAH3 2005-2023)
- **Downloaded on:** 2026-10-10 → `data/reference/pvgis/` (one city per region; settings in the CSV: 1 kWp, 14% system loss, 35° pitch, south-facing)

**Used for:** kWh generated per kWp per year by UK region in the Solar Finance estimate.

---

## 11. DESNZ UK Solar Roadmap (June 2025)

- **Document:** https://assets.publishing.service.gov.uk/media/685d6e483e6b7941f4e00afb/35.87_DESNZ_UK_Solar_Roadmap_final.pdf
- **Retrieved:** 2026-10-10 (cited, not stored)

**Used for:** the reference system size. Footnote 26 bases its savings on "a typical 3.5kW south-facing installation". `solar_system_size_kwp` in `config/emission_factors.csv`.

---

## 12. MCS MGD 003 Solar PV Self-Consumption, Issue 2.0 (April 2022)

- **Document:** https://mcscertified.com/wp-content/uploads/2022/04/MGD-003-Solar-PV-Self-Consumption-Issue-2.0-Final.pdf
- **Retrieved:** 2026-10-10 (cited, not stored)

**Used for:** the share of solar generation used at home, by annual generation and the customer's annual electricity use, with no battery. Occupancy is unknown, so the "in half the day" tables apply, as MGD 003 instructs. Only the 36 PV-only cells the 3.5 kWp design can reach are copied, with their table numbers, into `config/solar_self_consumption.csv`. Outside that range the estimate is unavailable.

---

## 13. Electrification of Heat Demonstration (Energy Systems Catapult, DESNZ-funded)

- **Report:** https://es.catapult.org.uk/wp-content/uploads/2023/03/EoH-Interim-Insights-from-Heat-Pump-Performance-Data-1.pdf
- **Retrieved:** 2026-10-10 (cited, not stored)

**Used for:** the median air-source heat pump seasonal performance factor (SPFH4 2.80; IQR 2.53-3.09; 291 homes) in the Green Home heat-pump scenario. `heat_pump_spf_ashp` in `config/emission_factors.csv`.

---

## 14. BEIS impact assessment: Improving boiler standards and efficiency (December 2022)

- **Document:** https://assets.publishing.service.gov.uk/media/63970d92d3bf7f3f84a7359c/Improving_boiler_standards_and_efficiency_consultation-stage_impact_assessment.pdf (IA BEIS067(C)-22-CHD)
- **Retrieved:** 2026-10-10 (cited, not stored)

**Used for:** the in-situ efficiency of an existing gas boiler (84%, paragraph 43), which converts the customer's gas use into heat demand in the heat-pump scenario. `gas_boiler_in_situ_efficiency` in `config/emission_factors.csv`.

---

## Data generated fully synthetically (no external source)

These must be documented with their generation rules in `docs/data.md`:

- **Paper statements and digital activity** — statement preference, app/online banking logins (Paperless Banking). The impact uses one A4 sheet per statement: no published page count was found, so this is a stated lower bound (`sheets_per_statement`).
- **Existing bank products** — vehicle loans, mortgages, investments
- **Hidden persona traits** — eco-mindedness, digital affinity, financial headroom, change readiness. They drive take-up but are never model features (D1).
- **Product take-up labels** — generated from documented rules over the features above, **plus the hidden traits and noise**, so relationships are plausible rather than arbitrary and held-out metrics stay realistic

---

## Citation format

When citing in reports or the dashboard, use:

> *Source: [Publisher], [Dataset title], [edition/year]. Retrieved [date] from [URL]. Contains public sector information licensed under the Open Government Licence v3.0.*
