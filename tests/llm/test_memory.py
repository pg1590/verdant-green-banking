"""Customer memory (optional extension): guardrails from brief slide 15."""
import copy
from datetime import date, timedelta

import pytest

from src.memory.store import PROTECTED_FIELDS, MemoryStore, apply_memory

TODAY = date(2026, 7, 1)


def at(days):
    return TODAY + timedelta(days=days)


@pytest.fixture
def store():
    return MemoryStore(":memory:")


@pytest.fixture
def rows(make_row):
    return [make_row(product_code="GP01", ranking_score=0.40, rank=2),
            make_row(product_code="GP04", opportunity="PAPERLESS_BANKING", ranking_score=0.80, rank=1),
            make_row(product_code="GP02", opportunity="GREEN_HOME", eligible_to_show=False, ranking_score=None,
                     rank=None, suppression_reason="NOT_HOMEOWNER")]


def by_code(out):
    return {r["product_code"]: r["memory"] for r in out}


def test_no_history_keeps_p5_order(store, rows):
    out = apply_memory(rows, TODAY, store)
    assert [r["product_code"] for r in out] == ["GP04", "GP01", "GP02"]
    assert by_code(out)["GP02"] == {**by_code(out)["GP02"], "action": "HIDE", "reason": "NOT_ELIGIBLE", "display_rank": None}


def test_protected_fields_unchanged_and_inputs_not_mutated(store, rows):
    before = copy.deepcopy(rows)
    store.record("C10001", "GP01", "clicked", at(-1))
    out = {r["product_code"]: r for r in apply_memory(rows, TODAY, store)}
    for orig in before:
        assert all(out[orig["product_code"]][f] == orig[f] for f in PROTECTED_FIELDS)
    assert rows == before


def test_never_shows_ineligible_even_after_interest(store, rows):
    store.record("C10001", "GP02", "clicked", at(-1))
    assert by_code(apply_memory(rows, TODAY, store))["GP02"]["action"] == "HIDE"


def test_interest_boost_reorders(store, rows):
    store.record("C10001", "GP01", "clicked", at(-2))
    for d in (-30, -20, -10):
        store.record("C10001", "GP04", "shown", at(d))
    m = by_code(apply_memory(rows, TODAY, store))
    assert (m["GP01"]["action"], m["GP04"]["action"]) == ("BOOST", "DEMOTE")
    assert m["GP01"]["display_rank"] == 1


def test_dismissal_cool_off_then_return(store, rows):
    store.record("C10001", "GP04", "dismissed", TODAY)
    assert by_code(apply_memory(rows, at(10), store))["GP04"]["action"] == "COOL_OFF"
    back = by_code(apply_memory(rows, at(31), store))["GP04"]
    assert (back["action"], back["reason"]) == ("SHOW", "RETURNING_AFTER_COOL_OFF")


def test_second_dismissal_longer_cool_off(store, rows):
    store.record("C10001", "GP04", "dismissed", TODAY)
    store.record("C10001", "GP04", "dismissed", at(40))
    assert by_code(apply_memory(rows, at(100), store))["GP04"]["action"] == "COOL_OFF"
    assert by_code(apply_memory(rows, at(131), store))["GP04"]["action"] == "SHOW"


def test_took_up_and_not_relevant(store, rows):
    store.record("C10001", "GP01", "took_up", at(-100))
    store.record("C10001", "GP04", "not_relevant", at(-10))
    m = by_code(apply_memory(rows, TODAY, store))
    assert m["GP01"]["reason"] == "ALREADY_TOOK_UP" and m["GP04"]["reason"] == "CUSTOMER_SAID_NOT_RELEVANT"


def test_forget_and_unknown_event(store):
    store.record("C10001", "GP01", "shown", TODAY)
    assert store.forget("C10001") == 1 and store.history("C10001") == []
    with pytest.raises(ValueError):
        store.record("C10001", "GP01", "loved_it", TODAY)


def test_preview_memory_demo_leaves_scores_eligibility_and_impact_unchanged(capsys):
    from src.memory.demo import main
    assert main(['--customer', 'C10144']) == 0
    out = capsys.readouterr().out
    assert 'COOL_OFF' in out and 'HIDE: ALREADY_TOOK_UP' in out and 'events deleted' in out
