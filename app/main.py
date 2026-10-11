"""Verdant: customer and portfolio journeys over the shared validated service."""
from __future__ import annotations
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st

from src.carbon.engine import UNAVAILABLE_REASONS
from src.recommend.runtime import load_runtime_service
from app import brief, charts, ui

st.set_page_config(page_title='Verdant · Green Customer Insights', page_icon='🌿', layout='wide', initial_sidebar_state='auto')
ui.install_style()
ui.install_motion()


@st.cache_resource
def service(path):
    backend = load_runtime_service(path)
    # Eligible-cohort averages are immutable for this loaded bundle and ignore feedback.
    means = {p['product_code']: p['cohort_mean_propensity'] for p in backend.opportunities()['products']}
    return backend, means


# Staff tool: the customer page previews what a customer would see, framed by staff-only detail.
PAGES = {'Home': 'home', 'Business dashboard': 'business', 'Customer view (preview)': 'customer'}


def navigate_workspace(view):
    st.session_state.view = view
    st.query_params['page'] = PAGES[view]


def workspace_changed():
    navigate_workspace(st.session_state.view)


def navigate_customer(cid):
    st.session_state['_active-customer'] = cid
    st.session_state.customer_id = cid
    navigate_workspace('Customer view (preview)')


def remember(key):
    st.session_state['_saved-' + key] = st.session_state[key]


def remember_customer():
    st.session_state['_active-customer'] = st.session_state.customer_id


def portfolio_navigation():
    navigate_workspace('Business dashboard')


def flash(text):
    st.session_state.feedback_flash = text


def clear_feedback(backend, cid):
    backend.forget_memory(cid)
    for key in list(st.session_state):
        if key.startswith(f'shown-{cid}-'):
            del st.session_state[key]
    flash('Your feedback history is cleared. Available opportunities are restored.')


@st.dialog('Clear feedback history?')
def confirm_clear_feedback(backend, cid):
    st.write('This deletes your saved interactions and feedback for this customer. Available opportunities can appear again.')
    st.caption('Your profile, model scores and impact estimates are preserved. Deleted feedback cannot be restored.')
    confirm, cancel = st.columns(2)
    if confirm.button('Clear history', type='primary', key='confirm-clear-feedback', width='stretch', on_click=clear_feedback, args=(backend, cid)):
        st.rerun()
    if cancel.button('Keep my history', key='cancel-clear-feedback', width='stretch'):
        st.rerun()


def greeting():
    hour = datetime.now().hour
    return 'Good morning' if hour < 12 else 'Good afternoon' if hour < 18 else 'Good evening'


def date_label(value):
    try:
        return datetime.strptime(value, '%Y-%m-%d').strftime('%d %b %Y')
    except (ValueError, TypeError):
        return 'Date unavailable'


def cohort_table(products):
    return pd.DataFrame([{
        'Product': p['product_name'], 'Relevant cohort': p['structurally_eligible'],
        'Shown now': p['displayed_opportunities'], 'High priority': p['high_priority'],
        'Median likelihood': p['propensity_median'], 'P90 likelihood': p['propensity_p90'],
        'Expected take-ups': p['expected_takeups'], 'With impact': p['impact_estimated'],
    } for p in products])


@st.dialog('Review your next step', width='large')
def review_opportunity(row, insight=None):
    st.subheader(row['product_name'])
    guidance = {
        'GP01': ['Confirm which vehicle you would replace and your current annual mileage.',
                 'Compare vehicle costs, charging access and finance terms with your relationship team.'],
        'GP02': ['Confirm the intended home upgrade and arrange a heat-loss or suitability assessment.',
                 'Compare installation costs, energy assumptions and finance terms with your relationship team.'],
        'GP03': ['Arrange a roof, shading and installation assessment.',
                 'Compare quotes, expected generation and finance terms with your relationship team.'],
        'GP04': ['Review whether digital statements meet your accessibility and record-keeping needs.',
                 'Use your bank’s authorised statement settings to make the change.'],
    }
    for i, text in enumerate(guidance[row['product_code']], 1):
        st.write(f'{i}. {text}')
    st.info('This review does not submit an application or change your account settings. Financial and installation decisions need the authorised checks.')
    pdf = brief.review_pdf(row, insight, guidance[row['product_code']])
    name = row['product_name'].replace(' ', '-')
    st.download_button('Download review brief (PDF)', pdf, file_name=f'Verdant-{name}-brief-{row["customer_id"]}.pdf',
                       mime='application/pdf', icon=':material/download:', type='primary', key='download-review')


def render_business(backend):
    summary = backend.summary()
    opportunities = backend.opportunities()
    segments = backend.segments()['segments']
    ui.hero('A clearer path to greener banking.',
            'Find the opportunities worth a conversation, understand the evidence, and explore their potential impact.',
            label='Welcome back · Portfolio workspace', emphasis='greener banking.')
    ui.product_states({code: ('shown' if count else 'unavailable', f'{count:,} opportunities') for code, count in summary['opportunities_by_product'].items()})
    cols = st.columns(4)
    for col, label, key in zip(cols,
        ['Customers analysed', 'With opportunities', 'Opportunities to review', 'Paperless opportunities'],
        ['customers_analysed', 'customers_with_opportunities', 'eligible_opportunities', 'digital_adoption_opportunities']):
        col.metric(label, f'{summary[key]:,}')
    ui.section('Potential annual impact', 'Estimates for the current cohort, with coverage shown for each measure.')
    cols = st.columns(3)
    for col, key, title, unit in zip(cols,
        ['kg_co2e_per_year', 'energy_kwh_per_year', 'paper_sheets_per_year'],
        ['CO₂e reduction', 'Energy saved', 'Paper saved'], ['kg CO₂e', 'kWh', 'sheets']):
        metric = summary['potential_impact'][key]
        value = f"{metric['value']:,.1f} {unit}" if metric['value'] is not None else 'Insufficient data'
        col.metric(title, value)
        col.caption(f"Estimate covers {metric['coverage_customers']:,} customers")
    with st.expander('How to read these estimates'):
        st.write(summary['impact_aggregation'])
        st.caption('Factor edition: ' + (', '.join(summary['factor_editions']) or 'No estimates available'))
        st.caption(summary['score_scope'])
        st.caption('Paperless CO₂e is separate: ' + (f"{summary['digital_impact']['kg_co2e_per_year']['value']:,.3f} kg/year" if summary['digital_impact']['kg_co2e_per_year']['value'] is not None else 'unavailable'))
    products_tab, regions_tab, trend_tab = st.tabs(['Product opportunities', 'Regions & customers', 'Portfolio trend'])
    with products_tab:
        with st.container(key='chart-products'):
            ui.section('Where to focus', 'Compare the relevant cohort, opportunities shown now and high-priority customers.')
            st.plotly_chart(charts.products_chart(opportunities['products']), config=charts.CONFIG, theme=None, use_container_width=True, key='product-chart')
        ui.section('Conversion opportunities', 'Modelled likelihood of take-up, rather than observed or incremental conversions.')
        frame = cohort_table(opportunities['products'])
        st.dataframe(frame, hide_index=True, width='stretch', column_config={
            'Median likelihood': st.column_config.NumberColumn(format='percent'),
            'P90 likelihood': st.column_config.NumberColumn(format='percent'),
            'Expected take-ups': st.column_config.NumberColumn(help=opportunities['definitions']['expected_takeups'], format='%.1f'),
            'Relevant cohort': st.column_config.NumberColumn(help=opportunities['definitions']['structurally_eligible']),
            'High priority': st.column_config.NumberColumn(help=opportunities['definitions']['high_priority']),
        })
        with st.expander('Definitions and scope'):
            for name, text in opportunities['definitions'].items():
                st.write(f'**{name.replace("_", " ").capitalize()}** — {text}')
        st.download_button('Download opportunity table', frame.to_csv(index=False), 'verdant-opportunities.csv', 'text/csv',
                           icon=':material/download:', key='download-opportunities')
        render_optional_summary(backend)
    with regions_tab:
        ui.section('Find a segment', 'Filters apply to this explorer. Portfolio totals above always describe the complete loaded cohort.')
        a, b = st.columns(2)
        region_options = ['All regions'] + sorted({item['region'] for item in segments})
        product_options = ['All products'] + list(ui.PRODUCTS)
        for key, options in [('region-filter', region_options), ('product-filter', product_options)]:
            if st.session_state.get(key) not in options:
                saved = st.session_state.get('_saved-' + key)
                st.session_state[key] = saved if saved in options else options[0]
        region = a.selectbox('Region', region_options, key='region-filter', on_change=remember, args=('region-filter',))
        product = b.selectbox('Product', product_options,
            format_func=lambda c: 'All products' if c == 'All products' else ui.PRODUCTS[c]['short'], key='product-filter', on_change=remember, args=('product-filter',))
        filtered = [s for s in segments if (region == 'All regions' or s['region'] == region) and
                    (product == 'All products' or s['product_code'] == product)]
        if filtered:
            with st.container(key='chart-regions'):
                st.caption('Bubble size represents customer/product opportunities. Click a bubble or use the segment selector below.')
                selected = st.plotly_chart(charts.regions_chart(filtered), config=charts.CONFIG, theme=None, use_container_width=True,
                    key='region-chart', on_select='rerun', selection_mode='points')
            options = [(s['region'], s['product_code']) for s in filtered]
            points = selected.selection.points
            # Map by custom data, never a stale point index after filtering.
            clicked = tuple(points[0]['customdata'][:2]) if points and points[0].get('customdata') else None
            if clicked != st.session_state.get('last-region-click'):
                st.session_state['last-region-click'] = clicked
                if clicked in options:
                    st.session_state['selected-segment'] = clicked
                    st.session_state['_saved-selected-segment'] = clicked
            if st.session_state.get('selected-segment') not in options:
                saved = st.session_state.get('_saved-selected-segment')
                st.session_state['selected-segment'] = saved if saved in options else options[0]
            segment_key = st.selectbox('Segment', options, key='selected-segment',
                format_func=lambda s: f'{s[0]} · {ui.PRODUCTS[s[1]]["short"]}', on_change=remember, args=('selected-segment',))
            segment = next(s for s in filtered if (s['region'], s['product_code']) == segment_key)
            with st.container(key='segment-panel'):
                st.subheader(f'{segment["region"]} · {ui.PRODUCTS[segment["product_code"]]["short"]}')
                st.write(f"**{segment['customers']} customers** · mean likelihood **{segment['mean_propensity']:.1%}** · mean confidence **{segment['mean_confidence']:.1%}**")
                if st.button('Explain this segment', icon=':material/auto_awesome:', key='explain-segment'):
                    with st.spinner('Preparing an explanation from this segment’s evidence…'):
                        try:
                            explanation = backend.business_insight(*segment_key)
                            st.markdown(ui.plain_markdown(explanation['summary']))
                            for bullet in explanation['bullets']:
                                st.markdown('• ' + ui.plain_markdown(bullet))
                            st.caption('Explanation source: ' + explanation['provenance']['text_source'])
                        except (ValueError, RuntimeError, OSError):
                            st.warning('A segment explanation is unavailable. You can still review the evidence and customers below.')
                if st.session_state.get('segment-customer') not in segment['customer_ids']:
                    st.session_state['segment-customer'] = segment['customer_ids'][0]
                cid = st.selectbox('Customer in segment', segment['customer_ids'], key='segment-customer')
                st.button('Preview customer view', icon=':material/arrow_forward:', type='primary',
                          on_click=navigate_customer, args=(cid,), key='open-customer')
        else:
            ui.empty_state('No opportunities in this selection', 'Choose a different region or product to explore the available customers.')
    with trend_tab:
        history = summary['portfolio_history']
        if len(history) > 1:
            with st.container(key='trend-panel'):
                ui.section('How opportunities change', 'Comparable dated snapshots of the same customer cohort.')
                mode = st.radio('Show trend for', ['Customers', 'Products'], horizontal=True, key='trend-mode')
                st.plotly_chart(charts.trend_chart(history, 'products' if mode == 'Products' else 'customers'),
                               config=charts.CONFIG, theme=None, use_container_width=True, key='trend-chart')
                st.caption('Frozen models rescore each dated snapshot. This is retrospective analysis, not historical deployment or measured conversions.')
                if history[0].get('window_start'):
                    st.caption('Each point uses its own 12-month window. Post-cutoff snapshots are display-only and are excluded from model evaluation.')
                st.dataframe(pd.DataFrame([{'Date': h['date'], 'Customers analysed': h.get('customers_analysed'),
                                           'With opportunities': h['customers_with_opportunities'],
                                           'Opportunities': h.get('eligible_opportunities')} for h in history]), hide_index=True, width='stretch')
        else:
            ui.empty_state('A trend needs more than one snapshot', 'Supply comparable dated portfolio snapshots to see how opportunities change.')


def render_profile(customer):
    a, b, c = st.columns(3)
    with a:
        st.markdown('**Your profile**')
        for label, field in [('Age', 'age'), ('Household size', 'household_size'), ('Home', 'home_status'),
                            ('Property', 'property_type'), ('EPC', 'epc_rating'), ('Heating', 'main_heating_fuel'),
                            ('Solar installed', 'has_solar'), ('Main vehicle', 'vehicle_type'), ('Vehicles', 'num_vehicles'),
                            ('Statements', 'statement_pref')]:
            value = customer.get(field)
            if isinstance(value, bool):
                value = 'Yes' if value else 'No'
            elif isinstance(value, str):
                value = value.replace('_', ' ').capitalize()
            st.caption(f'{label}: {value if value is not None else "Unknown"}')
    with b:
        st.markdown('**Products held**')
        held = customer.get('products_held')
        if held is None:
            held = [name for name, field in [('Mortgage', 'has_mortgage'), ('Vehicle loan', 'has_vehicle_loan'),
                    ('Savings', 'has_savings'), ('Investments', 'has_investments'), ('Credit card', 'has_credit_card')] if customer.get(field)]
        st.write(', '.join(h.replace('_', ' ').capitalize() for h in held) or 'None recorded')
        st.markdown('**Estimated annual energy & travel**')
        for label, field, unit in [('Electricity', 'est_electricity_kwh_year', 'kWh'),
                                  ('Gas', 'est_gas_kwh_year', 'kWh'), ('Road mileage', 'est_annual_km', 'km')]:
            value = customer.get(field)
            st.caption(f'{label}: ' + (f'{value:,.0f} {unit}' if value is not None else 'Unknown'))
    with c:
        st.markdown('**Activity in the feature window**')
        for label, field in [('Transactions', 'n_transactions_window'), ('Active months', 'active_months_window'),
                             ('Last transaction', 'last_transaction_date'), ('Paper statements', 'paper_statements_12m')]:
            if field in customer:
                st.caption(f'{label}: {customer[field] if customer[field] is not None else "Unknown"}')
        for label, field in [('Monthly spend', 'spend_total_m'), ('Fuel spend/month', 'spend_fuel_m'), ('App logins/month', 'avg_app_logins_m')]:
            value = customer.get(field)
            if value is not None:
                st.caption(f'{label}: {value:,.1f}')
        st.caption('Completeness: ' + (f'{customer["data_completeness"]:.0%}' if customer.get('data_completeness') is not None else 'Unknown'))
        st.caption('Data flags: ' + (customer.get('data_flags') or 'None'))
        st.caption('Conflicts: ' + (', '.join(customer.get('conflict_flags') or []) or 'None'))


def render_optional(backend, cid):
    """Optional products (Green Investments, Green Loans): rule-based signals, never a likelihood."""
    result = backend.optional_opportunities(cid)
    shown = [r for r in result['products'] if r['eligible_to_show']]
    if not shown:
        return
    ui.section('Other opportunities', 'Rule-based signals from your data. No likelihood is estimated for these products.')
    for row in shown:
        with st.container(border=True, key=f'optional-{cid}-{row["product_code"]}'):
            st.markdown(f'**{row["product_name"]}**')
            st.caption('Rule-based signal · not a model prediction')
            st.write(ui.plain_markdown(row['explanation']))
            why = UNAVAILABLE_REASONS.get(row['impact'].get('reason'))
            st.caption('No impact estimate: ' + (why[0].lower() + why[1:] if why else 'no documented method.'))
            with st.expander('Evidence'):
                st.json(row['evidence'])
            if st.button(row['action']['label'], key=f'optional-action-{cid}-{row["product_code"]}'):
                st.info('Next step: review this with your relationship team. Nothing has been applied for or changed.')
    st.caption(result['notice'])


def render_optional_summary(backend):
    summary = backend.optional_summary()
    ui.section('Optional opportunities', 'Green Investments and Green Loans, shown as rule-based signals without a likelihood model.')
    cols = st.columns(len(summary['products']))
    for col, product in zip(cols, summary['products']):
        with col:
            st.metric(product['product_name'], f'{product["eligible_customers"]:,} customers')
            signals = list(product['by_use_case'].items()) or list(product['by_signal'].items())[:3]
            st.caption(' · '.join(f'{name.replace("_", " ").lower()}: {count:,}' for name, count in signals))
    st.caption(summary['notice'])


def render_customer(backend, ids, cohort_means):
    st.button('Back to portfolio', icon=':material/arrow_back:', type='tertiary', on_click=portfolio_navigation, key='back-to-portfolio')
    active = st.session_state.get('_active-customer')
    st.session_state.customer_id = active if active in ids else ids[0]
    ui.staff_banner()
    cid = st.selectbox('Previewing customer · staff only', ids, key='customer_id', on_change=remember_customer, format_func=lambda x: f'{x} · {backend.customers[x].get("region", "Unknown region")}')
    customer = backend.customers[cid]
    current = backend.recommendations(cid)
    audit = backend.recommendations(cid, include_suppressed=True)['recommendations']
    ui.hero('Your next greener step.', 'Explore opportunities supported by your data. See the evidence, understand the estimate, and choose what matters to you.',
            label=f'{greeting()} · {cid} · {customer.get("region", "Unknown region")}', customer=True, emphasis='greener step.')
    st.caption(f'Data as of {date_label(customer.get("feature_cutoff_date", customer.get("data_as_of")))} · Display relevance is not lending approval')
    states = {}
    for row in sorted(audit, key=lambda r: r['product_code']):
        hidden = row.get('memory', {}).get('action') in {'HIDE', 'COOL_OFF'}
        states[row['product_code']] = ('unavailable', 'Not available') if not row['eligible_to_show'] else (
            ('hidden', 'Already taken up' if row.get('memory', {}).get('reason') == 'ALREADY_TOOK_UP' else 'Hidden by your feedback') if hidden else ('shown', 'Available to review'))
    ui.product_states(states)
    if st.session_state.get('feedback_flash'):
        ui.feedback_notice(st.session_state.pop('feedback_flash'))
    hidden_count = current['memory_hidden_count']
    if hidden_count:
        st.info(f'{hidden_count} {"opportunity is" if hidden_count == 1 else "opportunities are"} hidden by your feedback. Clear your feedback history below to restore available suggestions.')
    with st.expander('Customer 360 · staff only', icon=':material/lock:'):
        render_profile(customer)
    if current['recommendations']:
        ui.section('Opportunities to explore', 'Ordered by likelihood × confidence, then adjusted by your feedback.')
        options = [row['product_code'] for row in current['recommendations']]
        chooser_key = f'opportunity-{cid}'
        if st.session_state.get(chooser_key) not in options:
            st.session_state[chooser_key] = options[0]
        selected = st.radio('Choose an opportunity', options, key=chooser_key, horizontal=True,
            format_func=lambda code: ui.PRODUCTS[code]['short']) if len(options) > 1 else options[0]
        selected = selected if selected in options else options[0]
        # Count only insights actually presented in this session, avoiding fatigue on reruns.
        # Only explain the selected, visible opportunity; do not call a live model for unseen cards.
        with st.spinner('Preparing this opportunity…'):
            item = backend.insight(cid, product_code=selected)['insights'][0]
        row = item['recommendation']
        shown_key = f'shown-{cid}-{selected}'
        if backend.memory_store is not None and not st.session_state.get(shown_key):
            backend.record_event(cid, selected, 'shown')
            st.session_state[shown_key] = True
        with st.container(key='opportunity-detail'):
            ui.product_header(selected, row['product_name'], ui.PRODUCTS[selected]['detail'],
                              row.get('memory', {}).get('display_rank', row['rank']))
            # Provider text is plain Streamlit Markdown, never custom HTML or SVG.
            with st.container(key='insight-heading'):
                st.markdown(ui.plain_markdown(item['headline']))
            with st.container(key='insight-copy'):
                st.markdown(ui.plain_markdown(item['text']))
            if item['fallback_reason'] == 'INVALID_PROVIDER_OUTPUT':
                st.warning('A standard explanation is shown because the generated response could not be validated.')
            ui.staff_tag('Model scores · staff only')
            a, b, c = st.columns(3)
            a.metric('Likelihood of take-up', f'{row["propensity"]:.0%}', help='Calibrated model probability of take-up in the label window. It is not affordability, approval or campaign uplift.')
            b.metric('Confidence', f'{row["confidence"]:.0%}', help='Reliability score based on available data and model stability. It is not a probability of approval.')
            c.metric('Priority score', f'{row["ranking_score"]:.3f}', help='Original likelihood × confidence, before feedback changes display order.')
            mean = cohort_means.get(selected)
            st.plotly_chart(charts.likelihood_chart(row['propensity'], mean), config=charts.CONFIG, theme=None, use_container_width=True, key=f'likelihood-{cid}-{selected}')
            st.caption(f'Marker: {mean:.1%} average likelihood for the product’s relevant cohort.' if mean is not None else 'No comparable product-cohort average is available.')
            st.markdown('**Why you’re seeing this**')
            st.write(' · '.join(code.replace('_',' ').capitalize() for code in row['reason_codes']))
            impact = row['impact']
            if impact['status'] == 'ESTIMATED':
                metrics = [(impact[field], unit, title) for field, unit, title in [
                    ('kg_co2e_per_year','kg CO₂e/year','Potential emissions avoided'),
                    ('energy_kwh_per_year','kWh/year','Less purchased energy'),
                    ('paper_sheets_per_year','sheets/year','Paper saved')] if impact[field] is not None]
                cols = st.columns(len(metrics))
                for col, (value, unit, title) in zip(cols, metrics):
                    with col:
                        ui.impact_tile(value, unit, title)
                with st.expander('How this impact is estimated'):
                    st.write(impact['method'])
                    st.caption(f'Factors: {impact["factor_edition"]} · Horizon: {impact["horizon_years"]} year(s)')
                    for assumption in impact['assumptions']:
                        st.write('• ' + assumption)
            else:
                why = UNAVAILABLE_REASONS.get(impact.get('reason'))
                st.info(('No impact estimate is available: ' + why[0].lower() + why[1:] if why else
                         'No impact estimate is available for this case. More baseline or design information is needed.')
                        + ' No saving has been assumed.')
            with st.expander('Evidence and model provenance · staff only', icon=':material/lock:'):
                st.json(row['evidence'])
                st.caption(f'Model: {row["model_version"]} · Policy: {row["policy_version"]} · Data: {row["data_timestamp"]}')
                if item.get('provenance'):
                    st.caption('Explanation source: ' + item['provenance']['text_source'])
                if row['data_flags']:
                    st.caption('Data quality: ' + row['data_flags'])
            buttons = st.columns([2,1,1.2])
            if buttons[0].button(row['action']['label'], key=f'action-{cid}-{selected}', type='primary',
                                 icon=f':material/{ui.PRODUCTS[selected]["icon"]}:', width='stretch'):
                if backend.memory_store is not None:
                    backend.record_event(cid, selected, 'clicked')
                review_opportunity(row, item)
            for col, label, event in zip(buttons[1:], ['Not now', 'Not relevant to me'], ['dismissed','not_relevant']):
                if col.button(label, key=f'{event}-{cid}-{selected}', width='stretch', disabled=backend.memory_store is None):
                    backend.record_event(cid, selected, event)
                    from src.memory.store import COOL_OFF_DAYS
                    prior_dismissals = row.get('memory', {}).get('state', {}).get('dismissals', 0)
                    period = COOL_OFF_DAYS['not_relevant'] if event == 'not_relevant' else COOL_OFF_DAYS['dismissed_twice' if prior_dismissals >= 1 else 'dismissed']
                    flash(f'{row["product_name"]} is hidden for {period} days. Clear your feedback history to see it again.')
                    st.rerun()
            reason = row.get('memory',{}).get('reason')
            if reason and reason != 'NO_MEMORY_EFFECT':
                st.caption('Feedback preference: ' + reason.replace('_',' ').lower())
    else:
        ui.empty_state('No opportunities to show right now', 'Your current data or feedback limits the available suggestions. Review the profile and reasons below, or clear feedback to restore eligible opportunities.')
    render_optional(backend, cid)
    with st.expander('Why other products are not shown · staff only', icon=':material/lock:'):
        suppressed = [r for r in audit if not r['eligible_to_show']]
        if not suppressed:
            st.caption('No product is excluded by the current data rules. Feedback-hidden products are shown in the status strip above.')
        for row in suppressed:
            st.markdown(f'**{row["product_name"]}**')
            for code in row['suppression_reasons']:
                st.write('• ' + ui.friendly_code(code))
            st.caption('Audit codes: ' + ', '.join(row['suppression_reasons']))
    with st.expander('Customer snapshot · staff only', icon=':material/lock:'):
        st.json(customer)
    if backend.memory_store is not None:
        if st.button('Clear my feedback history', key='clear-feedback', icon=':material/history:',
                     disabled=not backend.memory_store.history(cid), help='Review and confirm deletion of this customer’s saved interactions. Original scores and estimates are preserved.'):
            confirm_clear_feedback(backend, cid)


def render_landing():
    with st.container(key='landing-page'):
        # Play the arrival sequence once per browser session, not on every rerun.
        if not st.session_state.get('_intro-seen'):
            st.session_state['_intro-seen'] = True
            ui.intro_splash()
        ui.landing_header()
        copy, illustration = st.columns([1.1, 1], gap='large', vertical_alignment='center')
        with copy:
            ui.landing_copy()
            st.button('Open business dashboard', key='landing-business', type='primary',
                      icon=':material/arrow_forward:', width='stretch',
                      on_click=navigate_workspace, args=('Business dashboard',))
            st.button('Preview a customer view', key='landing-customer', width='stretch',
                      on_click=navigate_workspace, args=('Customer view (preview)',))
        with illustration:
            ui.landing_art()
        ui.landing_principles()
        with st.expander('How Verdant works'):
            st.write('Verdant is a tool for bank teams. The business dashboard shows where green products fit across the portfolio. The customer view previews what an individual customer would be shown, with staff-only evidence alongside it.')
            st.caption('Relationship managers can review a customer’s opportunities, the evidence and estimated impact, then discuss a next step with them. Product relevance is never a credit decision.')
        ui.html('<footer class="landing-footer">Interactive preview using synthetic customer data. Impact figures are estimates. Product relevance is not credit approval.</footer>')


# URLs preserve the workspace on refresh; Home can render without loading customer data.
route = next((name for name, slug in PAGES.items() if slug == st.query_params.get('page', 'home')), 'Home')
st.session_state.view = route
if route == 'Home':
    render_landing()
    st.stop()


try:
    backend, cohort_means = service(os.getenv('GREEN_BUNDLE_PATH'))
except (OSError, ValueError, KeyError, TypeError):
    st.error('Customer data could not be loaded. Check the selected data bundle or rebuild the pipeline, then reload.')
    st.stop()
ids = list(backend.customers)
if not ids:
    st.info('No customers are available in this data bundle. Load a populated bundle to continue.')
    st.stop()
if st.session_state.get('customer_id') not in ids:
    st.session_state.customer_id = ids[0]
with st.sidebar:
    ui.brand()
    view = st.radio('Workspace', list(PAGES), key='view', on_change=workspace_changed)
    st.divider()
    st.caption(f'{len(ids):,} customers · ' + ('Preview cohort' if backend.bundle.get('bundle_scope') == 'PREVIEW' else 'Loaded cohort'))
    st.caption('Synthetic customer data')
    st.caption('Evidence-led suggestions. Estimates are potential, not achieved savings.')
    with st.expander('How it works'):
        st.write('1. Review the customer profile.\n2. Explore relevant opportunities.\n3. Inspect the evidence and estimated impact.\n4. Discuss the next step or give feedback.')
if backend.bundle.get('bundle_scope') == 'PREVIEW':
    st.caption(f'Preview · {len(ids):,} synthetic customers scored using models trained on the full population. Portfolio totals describe this preview.')
if backend.mode == 'DEMO_MOCK':
    st.warning('Demo fixtures: scores are illustrative and are not trained predictions.')
if view == 'Business dashboard':
    render_business(backend)
else:
    render_customer(backend,ids,cohort_means)
st.divider()
st.caption(backend.catalogue['policy_notice'])
ui.html('<div class="app-footer">Verdant · Green Customer Insights · Decisions remain with you and your authorised relationship team.</div>')
