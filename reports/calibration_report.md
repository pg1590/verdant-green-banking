# Calibration report

Generated from `data/raw` (10,000 customers, 4,859,579 transactions) vs official targets in `config/calibration_targets.yaml`. Shares are of all customers unless stated; record fields (with injected missing values) are measured on non-missing rows.

## Shares

| Attribute | Value | Target | Generated | Difference |
| --- | --- | --- | --- | --- |
| region | North East | 4.1% | 4.2% | +0.1% |
| region | North West | 11.2% | 10.6% | -0.7% |
| region | Yorkshire and The Humber | 8.1% | 7.8% | -0.3% |
| region | East Midlands | 7.3% | 7.5% | +0.2% |
| region | West Midlands | 8.7% | 8.9% | +0.2% |
| region | East of England | 9.3% | 9.1% | -0.2% |
| region | London | 12.7% | 12.8% | +0.1% |
| region | South East | 13.6% | 13.1% | -0.5% |
| region | South West | 8.8% | 9.1% | +0.3% |
| region | Wales | 4.8% | 5.2% | +0.4% |
| region | Scotland | 8.7% | 8.8% | +0.1% |
| region | Northern Ireland | 2.6% | 2.8% | +0.2% |
| urban_rural | urban | 14.6% | 14.3% | -0.3% |
| urban_rural | town | 68.0% | 68.5% | +0.5% |
| urban_rural | rural | 17.4% | 17.2% | -0.2% |
| household_size | 1 | 29.5% | 29.6% | +0.0% |
| household_size | 2 | 34.1% | 34.1% | -0.0% |
| household_size | 3 | 15.7% | 15.7% | +0.1% |
| household_size | 4 | 14.5% | 14.8% | +0.3% |
| household_size | 5 | 6.1% | 5.8% | -0.3% |
| age band | 18-24 | 2.6% | 2.6% | -0.1% |
| age band | 25-34 | 14.8% | 14.6% | -0.2% |
| age band | 35-44 | 17.5% | 17.8% | +0.2% |
| age band | 45-54 | 16.8% | 16.2% | -0.6% |
| age band | 55-64 | 18.3% | 18.5% | +0.2% |
| age band | 65-74 | 14.3% | 14.5% | +0.2% |
| age band | 75+ | 15.6% | 15.9% | +0.3% |
| home_status | owner_outright | 35.5% | 36.1% | +0.6% |
| home_status | owner_mortgage | 29.3% | 29.5% | +0.3% |
| home_status | renter | 35.2% | 34.3% | -0.8% |
| income_band | Q1 | 20.0% | 19.5% | -0.5% |
| income_band | Q2 | 20.0% | 20.7% | +0.7% |
| income_band | Q3 | 20.0% | 19.7% | -0.4% |
| income_band | Q4 | 20.0% | 20.3% | +0.3% |
| income_band | Q5 | 20.0% | 19.8% | -0.2% |
| property_type | detached | 17.9% | 17.7% | -0.2% |
| property_type | semi_detached | 24.6% | 24.5% | -0.1% |
| property_type | terraced | 27.7% | 28.0% | +0.3% |
| property_type | bungalow | 7.7% | 8.1% | +0.4% |
| property_type | flat | 22.0% | 21.8% | -0.3% |
| epc_rating | B | 4.2% | 4.5% | +0.3% |
| epc_rating | C | 52.0% | 52.3% | +0.3% |
| epc_rating | D | 34.7% | 34.0% | -0.8% |
| epc_rating | E | 6.3% | 6.5% | +0.2% |
| epc_rating | F | 2.1% | 2.3% | +0.2% |
| epc_rating | G | 0.6% | 0.5% | -0.1% |
| main_heating_fuel | gas | 85.1% | 85.0% | -0.1% |
| main_heating_fuel | electric | 9.0% | 9.1% | +0.1% |
| main_heating_fuel | oil | 3.2% | 3.0% | -0.1% |
| main_heating_fuel | other | 2.7% | 2.9% | +0.2% |
| num_vehicles | 0 | 24.9% | 25.2% | +0.3% |
| num_vehicles | 1 | 42.2% | 41.5% | -0.6% |
| num_vehicles | 2 | 33.0% | 33.3% | +0.3% |
| vehicle_type (car households) | petrol | 53.8% | 53.8% | -0.0% |
| vehicle_type (car households) | diesel | 28.2% | 28.0% | -0.1% |
| vehicle_type (car households) | hybrid | 12.3% | 12.5% | +0.2% |
| vehicle_type (car households) | ev | 5.8% | 5.7% | -0.0% |
| has_solar | true | 6.1% | 5.8% | -0.3% |

## Spend per household per month (all households, feature window)

| Category | ONS target (GBP) | Generated (GBP) | Ratio |
| --- | --- | --- | --- |
| Electricity (incl. dual-fuel share) | 86.67 | 85.82 | 0.99 |
| Gas (incl. dual-fuel share) | 61.10 | 61.89 | 1.01 |
| Road fuel | 79.73 | 79.73 | 1.00 |
| Public transport | 22.97 | 22.96 | 1.00 |
| Home improvement | 51.57 | 51.54 | 1.00 |
| Automotive | 129.13 | 129.14 | 1.00 |

## Mileage check

Mean implied miles per car (combustion-car households, from road-fuel spend): **5,290** (median 4,866) vs NTS0901 average **7,100** (spend is calibrated to ONS, so this is a consistency check, not a target).

## Personas

| Persona | Share |
| --- | --- |
| commuter_family | 25.4% |
| urban_renter | 20.0% |
| retired_owner | 18.0% |
| stretched_household | 15.0% |
| rural_owner | 11.7% |
| eco_professional | 9.8% |
