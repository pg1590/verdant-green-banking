# Optional extension: customer memory

Code: [src/memory/store.py](../src/memory/store.py). Tests: [tests/llm/test_memory.py](../tests/llm/test_memory.py). Demo: `python -m src.memory.demo`.

## What it does

Each customer has a persistent event log (SQLite, `data/memory/customer_memory.db`, git-ignored): which insights were shown and how the customer responded (`shown`, `clicked`, `dismissed`, `not_relevant`, `took_up`). On the next visit, memory adapts the **order and visibility** of P5's rows:

| Memory state | `memory.action` | `memory.reason` |
|---|---|---|
| Not eligible (P5) | HIDE | `NOT_ELIGIBLE` (memory never overrides eligibility) |
| Customer started the product | HIDE | `ALREADY_TOOK_UP` |
| Said "not relevant" (last 180 days) | COOL_OFF | `CUSTOMER_SAID_NOT_RELEVANT` |
| Dismissed (30 days; 90 days after a second dismissal) | COOL_OFF | `DISMISSED_COOL_OFF` |
| Clicked (last 30 days) | BOOST ×1.25 | `RECENT_INTEREST` |
| Shown 3+ times since the last click | DEMOTE ×0.5 | `SHOWN_WITHOUT_ENGAGEMENT` |
| Back after a cool-off | SHOW | `RETURNING_AFTER_COOL_OFF` |

The display score is P5's `ranking_score` × the memory multiplier. The result goes in `row["memory"]["display_rank"]`; **P5's `rank` and every other P5 field are left untouched**.

## Guardrails (brief, slide 15)

- Propensity, confidence, eligibility, impact, reason codes, evidence, `rank`, `ranking_score` and the suppression reason are never modified. A test checks this, and that the inputs aren't mutated.
- Memory can hide, cool off or demote an eligible insight. It can **never** show an ineligible one.
- The LLM never sees memory.
- Every change is explained with `memory.action` and `memory.reason`, plus the full memory state (FR7).
- `MemoryStore.forget(customer_id)` deletes everything stored for a customer (right to erasure).

## Integration (P5)

The shared runtime installs a MemoryStore in both API and Streamlit. Call
`service.record_event(cid, code, event)` on interactions and
`service.forget_memory(cid)` for erasure. `service.recommendations()` applies
feedback automatically; do not apply it a second time in the UI. The stored
P5 rank is immutable; `memory.display_rank` controls presentation. Default
databases include the bundle fingerprint so rebuilt synthetic identities do
not inherit previous events. GREEN_MEMORY_PATH provides an explicit override.

## Demo (trained preview, customer C10144)

```bash
python -m src.memory.demo                               # committed preview, C10144 (all four products eligible)
python -m src.memory.demo --bundle data/processed/team_bundle.json --customer C90001
```

The demo goes through P5's `RecommendationService` (the same path as the API and UI) with an
in-memory store, so it never touches the local database. It replays dated sessions with
`recommendations(..., on=date)`:

```
Day 0:  all four shown                       -> dismisses the 2nd, clicks the 1st
Day 7:  1st BOOST (recent interest); 2nd COOL_OFF (dismissed)        -> says the 2nd is not relevant
Day 45: 2nd still COOL_OFF (not relevant, 180 days)                  -> takes up the 1st
Day 60: 1st HIDE (already took up); unclicked rows DEMOTE (shown 3+ times)
Erase:  all events deleted; all four shown again in the original order
Check:  "Propensity, confidence, eligibility, impact and P5 rank unchanged: yes"
```

The command exits with an error if any protected field changed. `tests/llm/test_memory.py` runs it.
