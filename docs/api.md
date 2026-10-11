# API walkthrough

Start the API with `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`. `/docs` exposes the interactive OpenAPI UI. The default runtime uses the trained preview; `GREEN_BUNDLE_PATH` selects an explicit bundle.

## Read-only endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Runtime mode, preview scope and policy version |
| GET | `/customers` | Synthetic customer profiles |
| GET | `/customers/{id}/recommendations` | Eligible, ranked product opportunities |
| GET | `/customers/{id}/insight` | Validated explanation headline/text and provenance |
| GET | `/portfolio/summary` | Portfolio counts, estimated impact and trend |
| GET | `/portfolio/opportunities` | Product cohorts and conversion opportunities |
| GET | `/segments` | Regional/product customer groups |
| GET | `/segments/insight?region=London&product_code=GP03` | Grounded segment explanation |
| GET | `/portfolio/engagement` | Aggregate feedback activity |
| GET | `/customers/{id}/optional` | Rule-based optional opportunities |
| GET | `/portfolio/optional` | Optional-product aggregates |

These examples use the committed synthetic preview:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/customers/C10144/recommendations
curl http://localhost:8000/portfolio/summary
```

## Simulate changed circumstances

```bash
curl -X POST http://localhost:8000/customers/C90001/what-if \
  -H 'Content-Type: application/json' \
  -d '{"changes":{"vehicle_type":"none"}}'
```

The customer no longer has the recorded EV replacement use case. The response contains before/after results and `SIMULATION_NOT_SAVED`; the original customer stays unchanged. Unsupported profile fields are rejected.

## Ask the portfolio

```bash
curl -X POST http://localhost:8000/portfolio/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How many customers have solar opportunities?"}'
```

Questions are routed to computed facts; any generated wording is grounded in those facts. Unsupported questions fail rather than produce arbitrary SQL or invented figures. Providers, routing scope and offline behavior are described in [LLM design](llm_design.md).

## Feedback and erasure

```bash
curl -X POST http://localhost:8000/customers/C10144/events \
  -H 'Content-Type: application/json' \
  -d '{"product_code":"GP03","event":"dismissed"}'

curl -X DELETE http://localhost:8000/customers/C10144/memory
```

Events are `shown`, `clicked`, `dismissed`, `not_relevant` or `took_up`. Dismissal initially hides an opportunity for 30 days; repeated dismissal and other preferences use documented cool-off rules. Erasure removes feedback without changing the underlying profile, model scores or impact.

For automated experiments, set `GREEN_MEMORY_PATH=:memory:` or point it at a temporary SQLite database. This API currently has no production authentication and is intended for synthetic-data demonstrations.
