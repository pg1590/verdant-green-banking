"""Shared service for API and UI; explicit demo or team-output bundle, never silent fallback."""
import copy
import json
from datetime import date
from collections import defaultdict
from math import ceil
from pathlib import Path
from statistics import median
from threading import RLock
from typing import Callable

from .demo import demo_bundle
from .engine import build_recommendations, valid_date
from .eligibility import load_catalogue
from .validation import validate_team_customer, configured_factor_edition

# A displayed opportunity is high priority at 1.5x its product's mean eligible propensity.
HIGH_PRIORITY_LIFT = 1.5


class RecommendationService:
    def __init__(self, bundle: dict, insight_provider: Callable | None = None, memory_store=None):
        if bundle.get('schema_version') != '1.0':
            raise ValueError('Expected integration bundle schema_version 1.0')
        if bundle.get('data_mode') not in {'DEMO_MOCK', 'TEAM_OUTPUTS'}:
            raise ValueError('data_mode must explicitly be DEMO_MOCK or TEAM_OUTPUTS')
        self.mode = bundle['data_mode']
        if not isinstance(bundle.get('customers'), list):
            raise ValueError('customers must be a list')
        for field in ['predictions', 'impacts']:
            if not isinstance(bundle.get(field, {}), dict):
                raise ValueError(f'{field} must be a mapping keyed by customer ID')
            if any(not isinstance(value, dict) for value in bundle.get(field, {}).values()):
                raise ValueError(f'{field} customer entries must be product mappings')
        history = bundle.get('portfolio_history', [])
        if not isinstance(history, list):
            raise ValueError('portfolio_history must be a list')
        dates = []
        for snapshot in history:
            if not isinstance(snapshot, dict) or not valid_date(snapshot.get('date')):
                raise ValueError('History requires dated snapshot objects')
            count = snapshot.get('customers_with_opportunities')
            if type(count) is not int or count < 0:
                raise ValueError('History customer counts must be nonnegative integers')
            dates.append(snapshot['date'])
        if dates != sorted(set(dates)):
            raise ValueError('History dates must be unique and increasing')
        self.customers = {}
        for customer in bundle['customers']:
            if not isinstance(customer, dict):
                raise ValueError('Customer snapshots must be objects')
            cid = customer.get('customer_id')
            if not isinstance(cid, str) or not cid.strip() or cid in self.customers:
                raise ValueError('Customer IDs must be unique nonempty strings')
            if self.mode == 'TEAM_OUTPUTS':
                validate_team_customer(customer)
            self.customers[cid] = customer
        self.bundle = bundle
        self.catalogue = load_catalogue()
        self.insight_provider = insight_provider
        self.memory_store = memory_store
        self._aggregate_cache = {}
        self._aggregate_lock = RLock()
        self._rows = {}
        for cid, customer in self.customers.items():
            preds = bundle.get('predictions', {}).get(cid, {})
            if any(not isinstance(p, dict) for p in preds.values()):
                raise ValueError('Prediction product entries must be objects')
            if self.mode == 'TEAM_OUTPUTS' and any(p.get('data_mode') == 'DEMO_MOCK' or str(p.get('model_version', '')).startswith('mock_') for p in preds.values()):
                raise ValueError('Mock predictions cannot be loaded as TEAM_OUTPUTS')
            rows = build_recommendations(customer, preds, bundle.get('impacts', {}).get(cid, {}), self.catalogue)
            if self.mode == 'TEAM_OUTPUTS':
                raw_impacts = bundle.get('impacts', {}).get(cid, {})
                for row in rows:
                    raw = raw_impacts.get(row['product_code'])
                    if isinstance(raw, dict) and raw.get('status') == 'ESTIMATED' and row['impact']['status'] != 'ESTIMATED':
                        raise ValueError(f"{cid}/{row['product_code']}: malformed ESTIMATED impact; include valid metrics, method, assumptions, horizon and factor_edition or return INSUFFICIENT_DATA")
            for row in rows:
                row['data_mode'] = self.mode
            self._rows[cid] = rows
        editions = {r['impact']['factor_edition'] for rows in self._rows.values()
                    for r in rows if r['impact']['status'] == 'ESTIMATED'}
        if len(editions) > 1:
            raise ValueError('Mixed emissions factor editions: rebuild impacts using one configured edition')
        if self.mode == 'TEAM_OUTPUTS' and editions and editions != {configured_factor_edition()}:
            raise ValueError('Impact factor_edition must match settings.carbon.factor_edition')
        self.factor_editions = sorted(editions)
        # Strict JSON check prevents NaN/Infinity from leaking into API output.
        json.dumps(self._rows, allow_nan=False)

    @classmethod
    def demo(cls):
        return cls(demo_bundle())

    @classmethod
    def from_file(cls, path: str | Path):
        def invalid_constant(value):
            raise ValueError(f'Non-finite JSON number: {value}')
        return cls(json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid_constant))

    def recommendations(self, customer_id: str, include_suppressed: bool = False, on: date | None = None) -> dict:
        """`on` sets the memory date (default today); demos replay dated sessions with it."""
        if customer_id not in self.customers:
            raise KeyError(customer_id)
        rows = self._rows[customer_id]
        if self.memory_store is not None:
            from src.memory.store import apply_memory
            rows = apply_memory(rows, on or date.today(), self.memory_store)
        return {'customer_id': customer_id, 'data_mode': self.mode,
                'recommendations': [r for r in rows if include_suppressed or
                    (r['eligible_to_show'] and r.get('memory', {}).get('action') not in {'HIDE', 'COOL_OFF'})],
                'memory_hidden_count': sum(r['eligible_to_show'] and r.get('memory', {}).get('action') in {'HIDE', 'COOL_OFF'} for r in rows),
                'suppressed_count': sum(not r['eligible_to_show'] for r in rows),
                'policy_notice': self.catalogue['policy_notice']}

    def insight(self, customer_id: str, product_code: str | None = None) -> dict:
        if product_code is not None and product_code not in self.catalogue['products']:
            raise ValueError('Unknown product code')
        result = self.recommendations(customer_id)
        insights = []
        for row in result['recommendations']:
            if product_code is not None and row['product_code'] != product_code:
                continue
            # P4 injects a callable receiving ONLY the final structured object.
            # It must include its own validator/retry/fallback controls.
            explanation = self.insight_provider(row) if self.insight_provider else None
            headline = row['product_name']
            text = None
            if isinstance(explanation, dict):
                if all(isinstance(explanation.get(key), str) and explanation[key].strip()
                       for key in ('headline', 'text')):
                    headline, text = explanation['headline'], explanation['text']
            elif isinstance(explanation, str) and explanation.strip():
                # Retain compatibility with P4's earlier string interface.
                text = explanation
            fallback_reason = None
            if text is None:
                text = f"You may want to explore {row['product_name'].lower()}. Review the supporting evidence and any estimated impact."
                source = 'DETERMINISTIC_TEMPLATE'
                fallback_reason = 'INVALID_PROVIDER_OUTPUT' if self.insight_provider else 'MISSING_PROVIDER'
            else:
                source = 'P4_VALIDATED_PROVIDER'
            insights.append({'product_code': row['product_code'], 'headline': headline,
                             'text': text, 'source': source, 'fallback_reason': fallback_reason,
                             'provenance': explanation.get('provenance') if isinstance(explanation, dict) else None,
                             'recommendation': row})
        return {'customer_id': customer_id, 'data_mode': self.mode, 'insights': insights,
                'policy_notice': self.catalogue['policy_notice']}

    def record_event(self, customer_id: str, product_code: str, event: str, on: date | None = None) -> dict:
        if customer_id not in self.customers:
            raise KeyError(customer_id)
        if product_code not in self.catalogue['products']:
            raise ValueError('Unknown product code')
        if self.memory_store is None:
            raise ValueError('Customer memory is not configured')
        self.memory_store.record(customer_id, product_code, event, on or date.today())
        self._aggregate_cache.clear()
        return {'customer_id': customer_id, 'product_code': product_code, 'event': event, 'recorded': True}

    def forget_memory(self, customer_id: str) -> dict:
        if customer_id not in self.customers:
            raise KeyError(customer_id)
        if self.memory_store is None:
            raise ValueError('Customer memory is not configured')
        deleted = self.memory_store.forget(customer_id)
        self._aggregate_cache.clear()
        return {'customer_id': customer_id, 'deleted_events': deleted}

    def _cached(self, name: str, compute: Callable, copy_result: bool = True):
        """Portfolio views scan every customer (about 1 s at 10k), so they are cached until the
        date or the feedback store changes. stamp() also sees feedback written by another
        process sharing the database (the API and Streamlit)."""
        store = self.memory_store
        key = (date.today(), store.stamp() if store is not None and hasattr(store, 'stamp') else None)
        with self._aggregate_lock:  # concurrent first requests wait for one computation
            hit = self._aggregate_cache.get(name)
            if hit is None or hit[0] != key:
                hit = (key, compute())
                self._aggregate_cache[name] = hit
        return copy.deepcopy(hit[1]) if copy_result else hit[1]

    def _current(self) -> dict:
        """Shown rows per customer after feedback, shared by the portfolio views (read-only)."""
        return self._cached('current', lambda: {cid: self.recommendations(cid)['recommendations']
                                                for cid in self.customers}, copy_result=False)

    def summary(self) -> dict:
        return self._cached('summary', self._summary)

    def opportunities(self) -> dict:
        """FR6 conversion opportunities per product; displayed counts reconcile with summary()."""
        return self._cached('opportunities', self._opportunities)

    def segments(self) -> dict:
        return self._cached('segments', self._segments)

    def _summary(self) -> dict:
        current = list(self._current().values())
        eligible = [r for rows in current for r in rows]
        # Finance interventions compete; paperless is a separate service journey.
        finance = [next((r for r in sorted(rows, key=lambda r: r['rank']) if
                   self.catalogue['products'][r['product_code']]['kind'] == 'finance'), None)
                   for rows in current]
        finance = [r for r in finance if r is not None]
        estimated = [r for r in finance if r['impact']['status'] == 'ESTIMATED']
        paperless = [r for r in eligible if r['product_code'] == 'GP04' and r['impact']['status'] == 'ESTIMATED']
        def aggregate(rows, key):
            values = [r['impact'][key] for r in rows if r['impact'].get(key) is not None]
            return {'value': round(sum(values), 3) if values else None, 'coverage_customers': len(values)}
        metrics = {key: aggregate(estimated, key)
                   for key in ['kg_co2e_per_year', 'energy_kwh_per_year']}
        metrics['paper_sheets_per_year'] = aggregate(paperless, 'paper_sheets_per_year')
        digital = {'paper_sheets_per_year': metrics['paper_sheets_per_year'],
                   'kg_co2e_per_year': aggregate(paperless, 'kg_co2e_per_year')}
        customers_with_opportunities = len({r['customer_id'] for r in eligible})
        return {'data_mode': self.mode, 'customers_analysed': len(self.customers),
                'customer_split_counts': {split: sum((c.get('split') or 'unspecified') == split for c in self.customers.values())
                    for split in sorted({c.get('split') or 'unspecified' for c in self.customers.values()})},
                'score_scope': 'Synthetic population scores may include training customers; model performance is evaluated separately on held-out cohorts.',
                'customers_with_opportunities': customers_with_opportunities, 'eligible_opportunities': len(eligible),
                'opportunities_by_product': {code: sum(r['product_code'] == code for r in eligible) for code in self.catalogue['products']},
                'potential_impact': metrics,
                'impact_aggregation': 'CO2e/energy: one highest-ranked eligible finance opportunity per customer; no fallback for missing impact. Paperless sheets and CO2e are separate. Annual unweighted potential, not achieved impact.',
                'impact_coverage_customers': metrics['kg_co2e_per_year']['coverage_customers'],
                'finance_estimate_customers': len(estimated),
                'finance_opportunity_customers': len(finance), 'digital_impact': digital,
                'factor_editions': self.factor_editions,
                'digital_adoption_opportunities': sum(r['product_code'] == 'GP04' for r in eligible),
                'suppressed_opportunities': sum(not r['eligible_to_show'] for rows in self._rows.values() for r in rows),
                'data_timestamps': sorted({r['data_timestamp'] for r in eligible if r['data_timestamp']}),
                'portfolio_history': self.bundle.get('portfolio_history', []),
                'policy_notice': self.catalogue['policy_notice']}

    def _opportunities(self) -> dict:
        shown = [r for rows in self._current().values() for r in rows]
        audit = [r for rows in self._rows.values() for r in rows]
        products = []
        for code, product in self.catalogue['products'].items():
            cohort = [r['propensity'] for r in audit if r['product_code'] == code
                      and r['structurally_eligible'] and r['propensity'] is not None]
            values = sorted(r['propensity'] for r in shown if r['product_code'] == code)
            cohort_mean = sum(cohort) / len(cohort) if cohort else None
            threshold = round(HIGH_PRIORITY_LIFT * cohort_mean, 4) if cohort_mean else None
            products.append({
                'product_code': code, 'product_name': product['name'],
                'structurally_eligible': sum(r['product_code'] == code and r['structurally_eligible'] for r in audit),
                'displayed_opportunities': len(values),
                'high_priority': sum(v >= threshold for v in values) if threshold is not None else 0,
                'high_priority_threshold': threshold,
                'cohort_mean_propensity': round(cohort_mean, 4) if cohort_mean is not None else None,
                'propensity_median': round(median(values), 4) if values else None,
                'propensity_p90': round(values[max(0, ceil(0.9 * len(values)) - 1)], 4) if values else None,
                'expected_takeups': round(sum(values), 1),
                'impact_estimated': sum(r['product_code'] == code and r['impact']['status'] == 'ESTIMATED' for r in shown),
            })
        return {'data_mode': self.mode, 'products': products, 'definitions': {
            'structurally_eligible': 'Customers the catalogue allows for this product, before marketing consent, model validity and feedback.',
            'displayed_opportunities': 'Opportunities currently shown (same count as the summary).',
            'high_priority': f'Displayed customers with propensity at least {HIGH_PRIORITY_LIFT}x the mean propensity of the product\'s eligible cohort.',
            'expected_takeups': 'Sum of calibrated propensities over displayed customers: model-expected take-ups within the 3-month label window if every one were contacted. Not incremental uplift and not observed conversions.'}}

    def _segments(self) -> dict:
        groups = defaultdict(list)
        current = self._current()
        for cid, customer in self.customers.items():
            for row in current[cid]:
                if row['eligible_to_show']:
                    groups[(customer.get('region') or 'Unknown', row['product_code'])].append(row)
        rows = [{'region': region, 'product_code': code, 'customers': len(items),
                 'mean_propensity': round(sum(r['propensity'] for r in items) / len(items), 4),
                 'mean_confidence': round(sum(r['confidence'] for r in items) / len(items), 4),
                 'mean_ranking_score': round(sum(r['ranking_score'] for r in items) / len(items), 4),
                 'customer_ids': [r['customer_id'] for r in items]}
                for (region, code), items in groups.items()]
        rows.sort(key=lambda r: (-r['mean_ranking_score'], -r['customers'], r['region'], r['product_code']))
        return {'data_mode': self.mode, 'segments': rows, 'scope': 'Eligible display opportunities by region and product; propensity is not observed conversion.'}

    def ask(self, question: str, provider=None) -> dict:
        """Plain-English portfolio question: the LLM picks a fixed query, this service computes the facts."""
        from src.llm.ask import ask
        return ask(self, question, provider)

    def engagement(self) -> dict:
        """Feedback funnel per product from customer memory: shown -> clicked -> dismissed or not
        relevant -> took up (distinct customers). Recorded app interactions, not bank conversions."""
        return self._cached('engagement', self._engagement)

    def _engagement(self) -> dict:
        counts = self.memory_store.event_counts() if self.memory_store is not None and hasattr(
            self.memory_store, 'event_counts') else {}
        def rate(part, whole):
            return round(part / whole, 4) if whole else None
        products = []
        for code, product in self.catalogue['products'].items():
            c = counts.get(code, {})
            shown = c.get('shown', 0)
            declined = c.get('dismissed', 0) + c.get('not_relevant', 0)
            products.append({'product_code': code, 'product_name': product['name'], 'shown': shown,
                             'clicked': c.get('clicked', 0), 'dismissed': c.get('dismissed', 0),
                             'not_relevant': c.get('not_relevant', 0), 'took_up': c.get('took_up', 0),
                             'click_rate': rate(c.get('clicked', 0), shown), 'decline_rate': rate(declined, shown),
                             'take_up_rate': rate(c.get('took_up', 0), shown)})
        return {'data_mode': self.mode, 'memory_enabled': self.memory_store is not None, 'products': products,
                'scope': 'Distinct customers per event, recorded by the feedback memory of this prototype. '
                         'Demo interactions on synthetic customers, not observed bank conversions.'}

    def what_if(self, customer_id: str, changes: dict) -> dict:
        """Replay one customer with changed profile facts through rules, models and impact (not saved)."""
        from .what_if import what_if
        if customer_id not in self.customers:
            raise KeyError(customer_id)
        return what_if(self.customers[customer_id], self.bundle.get('predictions', {}).get(customer_id, {}),
                       self.bundle.get('impacts', {}).get(customer_id, {}), changes, self.catalogue)

    def optional_opportunities(self, customer_id: str) -> dict:
        """Optional products (Green Investments, Green Loans) as rule-based signals; never mixed
        into the ranked GP01-GP04 list and never given a propensity."""
        from .optional import evaluate, load_optional
        if customer_id not in self.customers:
            raise KeyError(customer_id)
        return {'customer_id': customer_id, 'data_mode': self.mode, 'products': evaluate(self.customers[customer_id]),
                'notice': load_optional()['notice']}

    def optional_summary(self) -> dict:
        return self._cached('optional', self._optional_summary)

    def _optional_summary(self) -> dict:
        from .optional import evaluate, load_optional
        cfg = load_optional()
        shown = [(c, r) for c in self.customers.values() for r in evaluate(c) if r['eligible_to_show']]
        products = []
        for code, p in cfg['products'].items():
            rows = [(c, r) for c, r in shown if r['product_code'] == code]
            regions = defaultdict(int)
            signals = defaultdict(int)
            for c, r in rows:
                regions[c.get('region') or 'Unknown'] += 1
                for s in r['reason_codes']:
                    signals[s] += 1
            products.append({'product_code': code, 'product_name': p['name'], 'eligible_customers': len(rows),
                             'by_use_case': {u: sum(u in r['use_cases'] for _, r in rows) for u in p.get('intended_use_cases', [])},
                             'by_signal': dict(sorted(signals.items(), key=lambda kv: -kv[1])),
                             'top_regions': sorted(regions.items(), key=lambda kv: (-kv[1], kv[0]))[:5]})
        return {'data_mode': self.mode, 'scope': cfg['scope'], 'notice': cfg['notice'], 'products': products}

    def business_insight(self, region: str, product_code: str) -> dict:
        from src.llm.explain import business_insight
        segment = next((s for s in self.segments()['segments']
                        if s['region'] == region and s['product_code'] == product_code), None)
        if segment is None:
            raise ValueError('No matching eligible segment')
        return business_insight({**segment, 'product_name': self.catalogue['products'][product_code]['name']})
