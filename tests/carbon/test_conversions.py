"""P2 derived physical quantities (feature-store columns est_annual_km, est_*_kwh_year). Owner: P2."""
import math

import pandas as pd
import pytest

from src.carbon import conversions as cv
from src.carbon.factors import get
from src.carbon.prices import window_energy_prices

PRICES = window_energy_prices()


def v(factor_id):
    return get(factor_id).value


def txn(customer, date, amount, category, ev=False, dual=False):
    return {"customer_id": customer, "date": pd.Timestamp(date), "amount": amount,
            "category": category, "is_ev_charging": ev, "is_dual_fuel": dual}


def months(customer, n, start="2025-07-15"):
    """One small 'Other' purchase per month so months_observed = n."""
    first_month = pd.Timestamp(start).to_period('M').to_timestamp()
    return [txn(customer, d, 5.0, "Other") for d in pd.date_range(first_month, periods=n, freq="MS") + pd.Timedelta(14, unit='D')]


@pytest.fixture
def derived():
    rows = (
        months("PETROL", 12) + [txn("PETROL", "2025-08-03", 1200.0, "Fuel")]
        # EV owner: charging is Fuel but must never become petrol mileage
        + months("EV", 12) + [txn("EV", "2025-09-01", 600.0, "Fuel", ev=True)]
        # hybrid household with both road fuel and charging
        + months("HYBRID", 12) + [txn("HYBRID", "2025-10-01", 600.0, "Fuel"),
                                  txn("HYBRID", "2025-10-02", 300.0, "Fuel", ev=True)]
        # single-fuel bills
        + months("BILLS", 12) + [txn("BILLS", "2026-01-05", 900.0, "Electricity"),
                                 txn("BILLS", "2026-01-05", 800.0, "Gas")]
        # one dual-fuel bill
        + months("DUAL", 12) + [txn("DUAL", "2026-01-05", 1500.0, "Utilities", dual=True),
                                txn("DUAL", "2026-01-06", 400.0, "Utilities")]  # water etc. ignored
        # six months of data only -> scaled x2; plus a txn after the cutoff (ignored)
        + months("HALF", 6) + [txn("HALF", "2025-08-01", 300.0, "Fuel"),
                               txn("HALF", "2026-08-01", 9999.0, "Fuel")]
        # bill smaller than a year of standing charges -> unknown, not zero
        + months("TINY", 12) + [txn("TINY", "2026-01-05", 50.0, "Electricity")]
    )
    customers = pd.DataFrame({
        "customer_id": ["PETROL", "EV", "HYBRID", "BILLS", "DUAL", "HALF", "TINY", "NODATA"],
        "vehicle_type": ["petrol", "ev", "hybrid", "none", "none", "diesel", "none", "petrol"],
    })
    out = cv.derive_quantities(pd.DataFrame(rows), customers)
    return out.set_index("customer_id")


def test_columns_cover_the_schema_fields(derived):
    from src.schemas.tables import FEATURES
    assert set(cv.DERIVED_COLUMNS) <= set(FEATURES)
    assert set(cv.DERIVED_COLUMNS) | set(cv.HELPER_COLUMNS) <= set(derived.columns)
    assert len(derived) == 8


def test_mileage_uses_road_fuel_only(derived):
    expected = 1200 / v("petrol_price_gbp_per_litre") * v("petrol_car_km_per_litre")
    assert derived.loc["PETROL", "est_annual_km"] == pytest.approx(expected)
    assert math.isnan(derived.loc["EV", "est_annual_km"])          # charging is not petrol
    assert derived.loc["EV", "road_fuel_spend_m"] == 0
    assert derived.loc["EV", "ev_charging_spend_m"] == pytest.approx(50)
    hybrid = 600 / v("petrol_price_gbp_per_litre") * v("hybrid_car_km_per_litre")
    assert derived.loc["HYBRID", "est_annual_km"] == pytest.approx(hybrid)  # the 300 charging excluded


def test_partial_year_is_annualised_and_window_respected(derived):
    assert derived.loc["HALF", "months_observed"] == 6
    expected = 300 * 2 / v("diesel_price_gbp_per_litre") * v("diesel_car_km_per_litre")
    assert derived.loc["HALF", "est_annual_km"] == pytest.approx(expected)


def test_single_fuel_bills_remove_the_standing_charge(derived):
    elec = (900 - PRICES["elec_standing_year"]) / PRICES["elec_unit"]
    gas = (800 - PRICES["gas_standing_year"]) / PRICES["gas_unit"]
    assert derived.loc["BILLS", "est_electricity_kwh_year"] == pytest.approx(elec)
    assert derived.loc["BILLS", "est_gas_kwh_year"] == pytest.approx(gas)
    assert not derived.loc["BILLS", "energy_from_dual_fuel"]


def test_dual_fuel_is_apportioned(derived):
    share = cv.dual_fuel_electricity_share()
    assert 0 < share < 1
    elec = (1500 * share - PRICES["elec_standing_year"]) / PRICES["elec_unit"]
    assert derived.loc["DUAL", "est_electricity_kwh_year"] == pytest.approx(elec)
    assert derived.loc["DUAL", "energy_from_dual_fuel"]
    # a medium-TDCV dual-fuel bill should come back as roughly medium TDCV usage
    medium_bill = (v("electricity_tdcv_medium_2023_kwh") * PRICES["elec_unit"]
                   + PRICES["elec_standing_year"]
                   + v("gas_tdcv_medium_2023_kwh") * PRICES["gas_unit"]
                   + PRICES["gas_standing_year"])
    rows = months("MED", 12) + [txn("MED", "2026-01-05", medium_bill, "Utilities", dual=True)]
    med = cv.derive_quantities(pd.DataFrame(rows), pd.DataFrame({"customer_id": ["MED"], "vehicle_type": ["none"]}))
    assert med.loc[0, "est_electricity_kwh_year"] == pytest.approx(v("electricity_tdcv_medium_2023_kwh"))
    assert med.loc[0, "est_gas_kwh_year"] == pytest.approx(v("gas_tdcv_medium_2023_kwh"))


def test_missing_data_stays_unknown(derived):
    assert math.isnan(derived.loc["TINY", "est_electricity_kwh_year"])  # below the standing charge
    no = derived.loc["NODATA"]
    assert no["months_observed"] == 0
    assert math.isnan(no["est_annual_km"]) and math.isnan(no["est_electricity_kwh_year"])
    assert not no["energy_from_dual_fuel"]
