# Contributing

Create a branch, install `requirements.txt`, and run the offline test suite before opening a pull request:

```bash
LLM_PROVIDER=offline PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

Keep scoring, eligibility, environmental arithmetic and explanation wording separated. Changes to impact estimates need documented factors, units, assumptions and hand-calculated examples. Never include real customer records, API keys, local feedback databases or generated raw data in a change. Preserve original source and licence notices.

UI changes should be checked at desktop/mobile sizes and with reduced motion enabled. Avoid rerunning the frozen held-out test evaluation as a routine development check; use validation and temporary fixtures.
