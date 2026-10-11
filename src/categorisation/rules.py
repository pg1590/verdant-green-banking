"""FR2 baseline: keyword + MCC rules classifier. Owner: P2.

Boundary decisions live in config/categories.yaml and docs/categorisation.md.
In the final categoriser (model.py) every rule that recognises the transaction (steps
1-5) decides; the model may only replace a step 6-7 fallback (ambiguous MCC, default Other).

    explain_rules(txns)    -> one row per txn: category, rule, strong, is_ev_charging, is_dual_fuel
    categorise_rules(txns) -> just the category (the baseline's prediction)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from src.common import config
from src.schemas.tables import CATEGORIES  # the only allowed outputs

RULE_COLUMNS = ("category", "rule", "strong", "is_ev_charging", "is_dual_fuel")

# Rule identifiers (also used as category_source by the model). Steps 1-4 are strong.
FORECOURT_SHOP = "FORECOURT_SHOP"
EV_CHARGING = "EV_CHARGING"
ENERGY_DUAL_FUEL = "ENERGY_DUAL_FUEL"
ENERGY_ELECTRICITY = "ENERGY_ELECTRICITY"
ENERGY_GAS = "ENERGY_GAS"
MCC_STRONG = "MCC_STRONG"
KEYWORD = "KEYWORD"
MCC_WEAK = "MCC_WEAK"
DEFAULT_OTHER = "DEFAULT_OTHER"
STRONG_RULES = frozenset({FORECOURT_SHOP, EV_CHARGING, ENERGY_DUAL_FUEL, ENERGY_ELECTRICITY,
                          ENERGY_GAS, MCC_STRONG})

_REF_TOKEN = re.compile(r"\b\w*\d\w*\d\w*\b")       # tokens with 2+ digits: store numbers, refs
_NON_WORD = re.compile(r"[^A-Z0-9& ]+")
_LETTER_AMP = re.compile(r"\b([A-Z])\s*&\s*([A-Z])\b")  # "B & Q" -> "B&Q"
_SPACES = re.compile(r"\s+")


def clean_text(text, noise: frozenset[str] = frozenset()) -> str:
    """Normalise merchant/description text so variants of one merchant look alike.

    "SHELL 4421 LEEDS" -> "SHELL LEEDS"; "Kwik-Fit (GB) Ltd" -> "KWIK FIT" (with noise words);
    "B & Q PLC" -> "B&Q"; "SAINSBURY'S PFS 0021" -> "SAINSBURYS PFS".
    """
    if not isinstance(text, str):
        return ""
    s = text.upper().replace("'", "").replace("’", "")
    s = _REF_TOKEN.sub(" ", s)
    s = _NON_WORD.sub(" ", s)
    s = _LETTER_AMP.sub(r"\1&\2", s)
    tokens = [t for t in s.split() if t not in noise and not t.isdigit()]
    return " ".join(tokens)


def _phrase_pattern(phrases: list[str]) -> re.Pattern | None:
    """Whole-word match of any phrase; longest phrases first so the longest match is found."""
    phrases = sorted({p for p in phrases if p}, key=len, reverse=True)
    if not phrases:
        return None
    alt = "|".join(re.escape(p) for p in phrases)
    return re.compile(rf"(?<![A-Z0-9&])(?:{alt})(?![A-Z0-9&])")


@dataclass(frozen=True)
class _RuleSet:
    noise: frozenset[str]
    shop_desc: re.Pattern | None
    ev_mcc: frozenset[int]
    ev_kw: re.Pattern | None
    energy_mcc: frozenset[int]
    suppliers: re.Pattern | None
    elec_hint: re.Pattern | None
    gas_hint: re.Pattern | None
    dual_kw: re.Pattern | None
    strong_mcc: dict[int, str]
    weak_mcc: dict[int, str]
    keyword_lists: tuple[tuple[str, re.Pattern | None, re.Pattern | None], ...]  # (cat, merchant, desc)
    priority: dict[str, int]


@lru_cache
def ruleset() -> _RuleSet:
    cfg = config.categories()
    noise = frozenset(cfg.get("merchant_noise", []))
    cats = cfg["categories"]
    if set(cats) != set(CATEGORIES):
        raise ValueError(f"categories.yaml must define exactly {CATEGORIES}")

    def merchant_pat(words):
        return _phrase_pattern([clean_text(w, noise) for w in words])

    def desc_pat(words):
        return _phrase_pattern([clean_text(w) for w in words])

    strong_mcc: dict[int, str] = {}
    for cat, spec in cats.items():
        for mcc in spec.get("mcc_hints", []):
            if mcc in strong_mcc:
                raise ValueError(f"MCC {mcc} listed under both {strong_mcc[mcc]} and {cat}")
            strong_mcc[int(mcc)] = cat
    weak_mcc = {int(k): v for k, v in cfg.get("weak_mcc", {}).items()}
    for cat in list(strong_mcc.values()) + list(weak_mcc.values()):
        if cat not in CATEGORIES:
            raise ValueError(f"unknown category {cat!r} in categories.yaml")

    energy = cfg["energy"]
    priority = {c: i for i, c in enumerate(cfg["keyword_priority"])}
    keyword_lists = tuple(
        (cat, merchant_pat(spec.get("keywords", [])), desc_pat(spec.get("description_keywords", [])))
        for cat, spec in cats.items()
    )
    return _RuleSet(
        noise=noise,
        shop_desc=desc_pat(cfg["forecourt_shop"]["description_keywords"]),
        ev_mcc=frozenset(int(m) for m in cfg["ev_charging"]["mcc"]),
        ev_kw=merchant_pat(cfg["ev_charging"]["keywords"]),
        energy_mcc=frozenset(int(m) for m in energy["mcc"]),
        suppliers=merchant_pat(energy["suppliers"]),
        elec_hint=desc_pat(energy["electricity_hints"]),
        gas_hint=desc_pat(energy["gas_hints"]),
        dual_kw=desc_pat(cfg["dual_fuel"]["keywords"]),
        strong_mcc=strong_mcc,
        weak_mcc=weak_mcc,
        keyword_lists=keyword_lists,
        priority=priority,
    )


def _search(pat: re.Pattern | None, text: str) -> bool:
    return bool(pat and pat.search(text))


def classify_one(merchant: str, description: str, mcc: int | None,
                 rs: _RuleSet | None = None) -> tuple[str, str, bool, bool]:
    """Apply the ordered rules to one transaction.

    Returns (category, rule, is_ev_charging, is_dual_fuel). Inputs are RAW strings.
    """
    rs = rs or ruleset()
    m = clean_text(merchant, rs.noise)
    d = clean_text(description)
    both = f"{m} {d}"

    # 1. Fuel-station shop purchase: a sandwich at "SHELL 4421 LEEDS" is not fuel.
    if _search(rs.shop_desc, d):
        return "Other", FORECOURT_SHOP, False, False

    # 2. EV charging: Fuel by team decision, flagged so road fuel can be separated.
    if (mcc is not None and mcc in rs.ev_mcc) or _search(rs.ev_kw, m):
        return "Fuel", EV_CHARGING, True, False

    # 3. Energy bills: fuel type from hints outside the supplier's own name.
    is_supplier = _search(rs.suppliers, m)
    if is_supplier or (mcc is not None and mcc in rs.energy_mcc):
        rest = rs.suppliers.sub(" ", m) if is_supplier else m
        hint_text = f"{rest} {d}"
        dual = _search(rs.dual_kw, both)
        elec = _search(rs.elec_hint, hint_text)
        gas = _search(rs.gas_hint, hint_text)
        if dual or (elec and gas):
            return "Utilities", ENERGY_DUAL_FUEL, False, True
        if elec:
            return "Electricity", ENERGY_ELECTRICITY, False, False
        if gas:
            return "Gas", ENERGY_GAS, False, False
        # no fuel hint: fall through (a boiler install from an energy brand is not a bill)
    elif _search(rs.dual_kw, both):
        return "Utilities", ENERGY_DUAL_FUEL, False, True

    # 4. Unambiguous MCC.
    if mcc is not None and mcc in rs.strong_mcc:
        return rs.strong_mcc[mcc], MCC_STRONG, False, False

    # 5. Keywords: longest matched phrase wins, ties by keyword_priority.
    best: tuple[int, int, str] | None = None
    for cat, mpat, dpat in rs.keyword_lists:
        for pat, text in ((mpat, m), (dpat, d)):
            hit = pat.search(text) if pat else None
            if hit:
                key = (-len(hit.group(0)), rs.priority.get(cat, len(rs.priority)), cat)
                best = key if best is None or key < best else best
    if best is not None:
        return best[2], KEYWORD, False, False

    # 6. Ambiguous MCC.
    if mcc is not None and mcc in rs.weak_mcc:
        return rs.weak_mcc[mcc], MCC_WEAK, False, False

    # 7. Everything else.
    return "Other", DEFAULT_OTHER, False, False


def group_codes(*keys) -> tuple[np.ndarray, np.ndarray]:
    """Group rows by equal-length key columns without building a key frame or merging.

    Returns the position of each group's first row and every row's group number; groups
    are numbered in order of first appearance. A missing value is a key of its own
    (factorize's default -1 sentinel would collide with another combined code).
    """
    codes = np.zeros(len(keys[0]) if keys else 0, dtype=np.int64)
    for key in keys:
        key_codes, uniques = pd.factorize(key, use_na_sentinel=False)
        # re-factorize after each key so the combined code stays below len(rows)**2
        codes, _ = pd.factorize(codes * len(uniques) + key_codes)
    _, first = np.unique(codes, return_index=True)
    return first, codes


def explain_rules(txns: pd.DataFrame) -> pd.DataFrame:
    """Rules output for every transaction, aligned to txns.index.

    Columns: category, rule, strong, is_ev_charging, is_dual_fuel. Rules are applied
    once per distinct (merchant_name, description, mcc) and copied back by position,
    so millions of rows are cheap.
    """
    rs = ruleset()
    mcc = txns["mcc"] if "mcc" in txns else pd.Series(pd.NA, index=txns.index, dtype="Int64")
    first, codes = group_codes(txns["merchant_name"], txns["description"], mcc)
    merchants = txns["merchant_name"].iloc[first].astype("string").fillna("")
    descriptions = txns["description"].iloc[first].astype("string").fillna("")
    results = pd.DataFrame(
        [classify_one(m, d, None if pd.isna(c) else int(c), rs)
         for m, d, c in zip(merchants, descriptions, mcc.iloc[first])],
        columns=["category", "rule", "is_ev_charging", "is_dual_fuel"])
    results["strong"] = results["rule"].isin(STRONG_RULES)
    out = results[list(RULE_COLUMNS)].take(codes)
    out.index = txns.index
    return out


def categorise_rules(txns: pd.DataFrame) -> pd.Series:
    """Return one category per row of `txns` (merchant_name, mcc, description).

    Never returns anything outside CATEGORIES; unknown -> "Other".
    """
    return explain_rules(txns)["category"].rename("category")


def is_ev_charging(txns: pd.DataFrame) -> pd.Series:
    """Compatibility helper for P1: separate charging from road fuel."""
    return explain_rules(txns)["is_ev_charging"]
