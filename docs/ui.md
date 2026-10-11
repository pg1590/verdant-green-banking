# Web interface (Verdant)

How the interface is organised and built. Moved from the README, which holds setup and run
instructions only.

Verdant opens on a minimal landing page with entry points to both workspaces.
Use **Home** in the sidebar to return. Workspace URLs (`?page=business` and
`?page=customer`) also work directly and survive refresh. The introduction loads
without requiring the customer bundle.

Verdant is a tool for bank teams (relationship, product and sustainability), with two
workspaces. **Business dashboard** compares product
cohorts, shows regional segments and dated trends, and exports an opportunity
CSV. Clicking a regional bubble selects its segment; **Preview customer view**
opens the chosen customer directly. Explorer filters do not change global totals.

**Customer view (preview)** shows what one customer would see, framed for bank staff: items marked *staff only* (customer selector, Customer 360, model scores, provenance) are never customer-facing. It shows product status, then one selected opportunity
with likelihood, confidence, evidence and estimated impact. The action opens a
review checklist and downloadable brief; it does not submit an application or
change account settings. Feedback uses the existing persistent cool-off rules.
Erasing feedback asks for confirmation and preserves the original scores/estimates.

Motion uses local CSS and the native Web Animations API: masked hero lines,
layered illustration depth, one-time scroll reveals and restrained hover feedback.
The controller observes Streamlit rerenders, cleans up removed nodes and respects
live `prefers-reduced-motion` changes. Native scrolling and widget callbacks are
preserved; no animation CDN or additional JavaScript package is required.

Plotly provides hoverable charts. Illustrations and licensed fonts are served
locally, so the visual interface needs no external asset service. Font licences
are included in `app/static/fonts/`. The preview and synthetic-data scope remain
visible. Use `LLM_PROVIDER=offline` for reproducible template-only explanations.
