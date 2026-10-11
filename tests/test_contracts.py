"""Cross-team contract tests: P1's schemas, P3's reason codes and P2's configs
agree with P5's catalogue, eligibility and engine."""

from src.common import config
from src.common.records import to_records
from src.data_gen.mock import generate_mock
from src.recommend import check_eligibility, eligible_customer_product_pairs, load_catalogue
from src.recommend.demo import demo_bundle
from src.schemas import tables as T


def test_reason_codes_reference_real_features_and_products():
    rc = config.reason_codes()["reason_codes"]
    for code, spec in rc.items():
        assert spec["feature"] in T.FEATURES, f"{code}: unknown feature {spec['feature']}"
        assert set(spec["products"]) <= set(T.PRODUCT_CODES), code
        assert spec["threshold"]["type"] in {"percentile", "absolute", "external", "flag"}, code
        assert spec["customer_text"], code


def test_every_product_has_reason_codes():
    rc = config.reason_codes()["reason_codes"]
    for code in T.PRODUCT_CODES:
        assert any(code in r["products"] for r in rc.values()), f"{code} has no reason codes"


def test_catalogue_matches_schema():
    cat = load_catalogue()
    for code, product in cat["products"].items():
        assert product["opportunity"] == T.OPPORTUNITY_BY_PRODUCT[code]
        for rule in product["rules"]:
            assert rule["field"] in T.FEATURES, f"{code}: rule reads {rule['field']}, not in feature store"


def test_demo_reason_codes_are_in_the_dictionary():
    rc = config.reason_codes()["reason_codes"]
    for products in demo_bundle()["predictions"].values():
        for pred in products.values():
            assert set(pred["reason_codes"]) <= set(rc), pred["reason_codes"]


def test_latent_traits_are_never_features():
    assert not set(T.HIDDEN_COLUMNS) & set(T.FEATURES)
    assert not set(T.HIDDEN_COLUMNS) & set(T.NON_MODEL_COLUMNS)


def test_mock_customers_flow_through_p5_eligibility():
    customers = to_records(generate_mock(80, seed=3)["customers"])
    assert all(type(c["marketing_consent"]) is bool for c in customers)
    pairs = eligible_customer_product_pairs(customers)
    assert len(pairs) == 4 * len(customers)
    shown = {code: sum(p["eligible_to_show"] for p in pairs if p["product_code"] == code) for code in T.PRODUCT_CODES}
    assert all(shown.values()), f"a product is never eligible: {shown}"
    for c in customers:
        if c["home_status"] == "renter":
            assert not check_eligibility(c, "GP03").eligible_to_show


def test_to_records_gives_engine_ready_dates_and_nulls():
    """Why to_records exists: raw pandas rows carry Timestamps and NaN, which
    P5's engine (ISO date strings) and JSON bundle (no NaN) reject."""
    raw = generate_mock(5, seed=3)["customers"].to_dict("records")[0]
    assert not isinstance(raw["data_as_of"], str)
    rec = to_records(generate_mock(5, seed=3)["customers"])[0]
    assert rec["data_as_of"] == str(config.settings()["calendar"]["feature_cutoff_date"])
    assert all(v == v for v in rec.values())  # no NaN


def test_demo_customers_use_canonical_region_and_income():
    for customer in demo_bundle()['customers']:
        assert customer['region'] in T.REGIONS
        assert customer['income_band'] in T.INCOME_BANDS
