import os

import pytest

# Tests must never call a paid or rate-limited API.
os.environ["LLM_PROVIDER"] = "offline"


@pytest.fixture
def make_row():
    """A final eligible row in src/recommend/engine.py's format, with an estimated impact."""
    def factory(**overrides):
        row = {
            "customer_id": "C10001", "opportunity": "EV_FINANCE", "product_code": "GP01",
            "product_name": "EV Finance", "propensity": 0.84, "confidence": 0.87, "model_confidence": 0.87,
            "reason_codes": ["HIGH_FUEL_SPEND", "HIGH_ANNUAL_MILEAGE", "EXISTING_VEHICLE"],
            "evidence": {"avg_monthly_fuel_spend_gbp": 210, "est_annual_km": 18400, "vehicle_type": "petrol"},
            "data_flags": "",
            "impact": {"kg_co2e_per_year": 1850, "energy_kwh_per_year": None, "paper_sheets_per_year": None,
                       "method": "Avoided emissions vs petrol baseline (GHG Protocol project method)",
                       "factor_edition": "DESNZ GHG Conversion Factors 2026",
                       "assumptions": ["Mileage estimated from fuel spend"], "horizon_years": 1, "status": "ESTIMATED"},
            "eligible_to_show": True, "structurally_eligible": True, "suppression_reason": None,
            "suppression_reasons": [], "data_timestamp": "2026-06-30", "model_version": "gp01_lgbm_v1",
            "policy_version": "v1", "eligibility_scope": "DISPLAY_ONLY_NOT_CREDIT_APPROVAL",
            "ranking_score": 0.7308, "rank": 1, "action": {"label": "Explore EV options", "intent": "explore_ev"},
            "data_mode": "TEAM_OUTPUTS",
        }
        row.update(overrides)
        return row
    return factory
