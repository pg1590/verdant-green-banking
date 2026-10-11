# Portfolio and resume notes

## Accurate positioning

**Verdant — Explainable Green Banking** is a collaborative applied-ML project with a portfolio edition maintained by Prakhar Gupta. Describe your role as integration/product engineering and distinguish personal implementation from the team's model, data and GenAI contributions.

A concise resume line:

> Built the integration and product layer for Verdant, an explainable green-banking prototype, connecting calibrated ML, eligibility rules, sourced impact estimates and grounded explanations through FastAPI and an interactive Streamlit dashboard.

Supporting bullets:

- Implemented a shared recommendation service and REST APIs for eligibility, ranking, portfolio analytics, customer feedback and what-if workflows over a trained 213-customer preview.
- Built a responsive dashboard and customer-view journey with interactive Plotly charts, motion/accessibility support and downloadable PDF review briefs.
- Integrated a reproducible synthetic-data pipeline designed for 10,000 customers, with scenario, contract, API and UI validation; documented fairness, uncertainty and missing-data behavior.

Use the recorded synthetic evaluation reports if discussing model performance. Do not describe estimated impact as achieved CO₂ savings, synthetic records as real customers, or team-wide ML/GenAI work as solely authored.

## Interview demo

1. Open the business dashboard and explain what counts as an eligible, shown or high-priority opportunity.
2. Select a region/product segment and preview a customer.
3. Explain the difference between take-up likelihood, confidence, eligibility and approval.
4. Review the source factors and assumptions behind an impact estimate; show an unavailable-data case.
5. Download the review brief and demonstrate feedback cool-off/erasure.
6. Use the what-if API to simulate a sold car; show that the original profile is unchanged.
7. Explain the shared-service architecture, cutoff/label separation, validator and cache boundaries.

## Scope

This is a working synthetic-data prototype, not a deployed bank system. Model and LLM evidence originate from the team experiment; the portfolio edition preserves those reports and licences while adding standalone presentation, setup documentation and CI.
