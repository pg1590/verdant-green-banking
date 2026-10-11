"""FR2 rules baseline: cleaning, boundary decisions and output contract. Owner: P2."""
import pandas as pd
import pytest

from src.categorisation import rules
from src.schemas.tables import CATEGORIES


@pytest.mark.parametrize("raw, expected", [
    ("SHELL 4421 LEEDS", "SHELL LEEDS"),
    ("Kwik-Fit (GB) Ltd", "KWIK FIT"),
    ("B & Q PLC", "B&Q"),
    ("SAINSBURY'S PFS 0021", "SAINSBURYS PFS"),
    ("AMZN Mktp UK*2X4KD", "AMZN MKTP"),
    ("TFL TRAVEL CH*2231", "TFL TRAVEL CH"),
    (None, ""),
])
def test_clean_merchant(raw, expected):
    assert rules.clean_text(raw, rules.ruleset().noise) == expected


# (merchant, description, mcc) -> (category, is_ev_charging, is_dual_fuel)
CASES = [
    # forecourt: pump vs shop vs car wash, same merchant text
    (("SHELL 4421 LEEDS", "CARD PAYMENT", 5541), ("Fuel", False, False)),
    (("SHELL 4421 LEEDS", "SHOP PURCHASE", 5541), ("Other", False, False)),
    (("SHELL 4421 LEEDS", "CONTACTLESS", 5499), ("Other", False, False)),
    (("BP HALL GREEN", "CARD PAYMENT", 7542), ("Automotive", False, False)),
    # EV charging is Fuel (team decision) and flagged, even under a fuel brand
    (("BP PULSE UK", "CARD PAYMENT", 5552), ("Fuel", True, False)),
    (("OCTOPUS EV CHARGE", "APP PAYMENT", None), ("Fuel", True, False)),
    (("POD POINT LTD", "CARD PAYMENT", None), ("Fuel", True, False)),
    # energy bills: fuel type from hints, not from the supplier's name
    (("BRITISH GAS", "DIRECT DEBIT DUAL FUEL", None), ("Utilities", False, True)),
    (("EDF ENERGY DUAL FUEL", "DD GAS & ELEC", 4900), ("Utilities", False, True)),
    (("BRITISH GAS", "DD GAS ACCOUNT", 4900), ("Gas", False, False)),
    (("SCOTTISHPOWER", "DIRECT DEBIT GAS A/C 7781", None), ("Gas", False, False)),
    (("OCTOPUS ENERGY ELEC", "DD ELECTRICITY", 4900), ("Electricity", False, False)),
    (("E.ON NEXT", "DIRECT DEBIT ELEC A/C 551200", None), ("Electricity", False, False)),
    (("BRITISH GAS SERVICES", "CARD PAYMENT", 1711), ("Home Improvement", False, False)),
    (("THAMES WATER", "DIRECT DEBIT", 4900), ("Utilities", False, False)),
    (("CERTAS ENERGY UK", "CARD PAYMENT", 5983), ("Utilities", False, False)),
    # MCC alone is not enough
    (("NETFLIX.COM", "CARD PAYMENT", 4899), ("Other", False, False)),
    (("VIRGIN MEDIA", "DIRECT DEBIT", 4899), ("Utilities", False, False)),
    (("HMRC SELF ASSESSMENT", "ONLINE", 9311), ("Other", False, False)),
    (("LEEDS CITY COUNCIL", "COUNCIL TAX", 9311), ("Utilities", False, False)),
    # D17 taxis, flights, furniture, loan repayments
    (("UBER *TRIP", "APP PAYMENT", 4121), ("Other", False, False)),
    (("EASYJET", "ONLINE PAYMENT", 4511), ("Other", False, False)),
    (("IKEA MANCHESTER", "CARD PAYMENT", 5712), ("Other", False, False)),
    (("BLACK HORSE LTD", "DIRECT DEBIT", None), ("Other", False, False)),
    # text-only (missing MCC)
    (("TESCO PFS 2291", "CARD PAYMENT", None), ("Fuel", False, False)),
    (("TESCO STORES 2291", "CARD PAYMENT", None), ("Other", False, False)),
    (("SCREWFIX DIRECT", "ONLINE", None), ("Home Improvement", False, False)),
    (("CAR INSURANCE DD", "DIRECT DEBIT", None), ("Automotive", False, False)),
    (("HEATING OIL DIRECT", "CARD PAYMENT", None), ("Utilities", False, False)),
    (("TRAINLINE.COM", "ONLINE", None), ("Public Transport", False, False)),
    (("SOME UNKNOWN SHOP", "CARD PAYMENT", None), ("Other", False, False)),
]


@pytest.mark.parametrize("txn, expected", CASES, ids=[c[0][0] + "|" + c[0][1] for c in CASES])
def test_boundary_decisions(txn, expected):
    category, rule, ev, dual = rules.classify_one(*txn)
    assert (category, ev, dual) == expected, rule


def test_ruleset_config_is_consistent():
    rs = rules.ruleset()
    assert set(rs.strong_mcc.values()) <= set(CATEGORIES)
    assert set(rs.weak_mcc.values()) <= set(CATEGORIES)
    assert not set(rs.strong_mcc) & set(rs.weak_mcc), "an MCC cannot be both strong and weak"


def test_explain_rules_aligns_with_any_index_and_duplicates():
    txns = pd.DataFrame({
        "merchant_name": ["SHELL 1 LEEDS", "NETFLIX.COM", "SHELL 1 LEEDS", None],
        "description": ["CARD PAYMENT", "CARD PAYMENT", "SHOP PURCHASE", None],
        "mcc": pd.array([5541, 4899, 5541, None], dtype="Int64"),
    }, index=[30, 10, 20, 99])
    out = rules.explain_rules(txns)
    assert list(out.index) == [30, 10, 20, 99]
    assert list(out.columns) == list(rules.RULE_COLUMNS)
    assert out["category"].tolist() == ["Fuel", "Other", "Other", "Other"]
    assert out.loc[30, "strong"] and not out.loc[99, "strong"]
    assert rules.categorise_rules(txns).isin(CATEGORIES).all()


def test_hand_labelled_fixture_only_uses_valid_categories():
    from src.categorisation.evaluate import load_hand_labelled
    hand = load_hand_labelled()
    assert hand["true_category"].isin(CATEGORIES).all()
    assert rules.categorise_rules(hand).isin(CATEGORIES).all()
