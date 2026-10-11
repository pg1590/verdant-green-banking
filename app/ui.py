"""Presentation primitives. No scores, eligibility or impact are calculated here."""
from __future__ import annotations

import base64
import hashlib
import re
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ASSETS = Path(__file__).parent / 'assets'
PRODUCTS = {
    'GP01': {'short': 'EV finance', 'asset': 'ev.svg', 'icon': 'electric_car', 'tag': 'Travel', 'detail': 'Explore a lower-emission journey.'},
    'GP02': {'short': 'Green home', 'asset': 'home.svg', 'icon': 'energy_savings_leaf', 'tag': 'Home', 'detail': 'Explore a more efficient home.'},
    'GP03': {'short': 'Solar finance', 'asset': 'solar.svg', 'icon': 'solar_power', 'tag': 'Energy', 'detail': 'Explore rooftop renewable energy.'},
    'GP04': {'short': 'Paperless', 'asset': 'paperless.svg', 'icon': 'description', 'tag': 'Everyday banking', 'detail': 'Keep statements digital and use less paper.'},
}


@lru_cache(maxsize=8)
def asset(name: str) -> str:
    encoded = base64.b64encode((ASSETS / name).read_bytes()).decode()
    return f'data:image/svg+xml;base64,{encoded}'


def html(value: str):
    st.html(value)


@lru_cache(maxsize=1)
def stylesheet(style_mtime):
    faces = []
    for file, family in [('atkinsonhyperlegiblenext.woff2', 'Atkinson'), ('schibstedgrotesk.woff2', 'Schibsted')]:
        faces.append(f"@font-face {{font-family:'{family}';src:url('app/static/fonts/{file}') format('woff2');font-weight:100 900;font-display:swap;}}")
    return '<style>' + ''.join(faces) + '\n' + (Path(__file__).parent / 'style.css').read_text() + '</style>'


def install_style():
    st.html(stylesheet((Path(__file__).parent / "style.css").stat().st_mtime_ns))


@lru_cache(maxsize=1)
def motion_script(script_mtime):
    script = (Path(__file__).parent / 'motion.js').read_text()
    return script.replace('__BUILD__', hashlib.sha256(script.encode()).hexdigest()[:16])


def install_motion():
    """An authored, same-origin controller; content stays usable without it.

    Streamlit 1.50 strips JavaScript from st.html. Its component iframe boots
    the controller in the host window, where it can observe native rerenders.
    No customer/provider text is interpolated into executable markup.
    """
    script = motion_script((Path(__file__).parent / 'motion.js').stat().st_mtime_ns)
    with st.container(key='motion-controller'):
        components.html('<!doctype html><html><body><script>' + script + '</script></body></html>',
                        height=0, scrolling=False, tab_index=-1)


def brand():
    html('<div class="brand"><span class="brand-mark" aria-hidden="true">v</span><div>verdant<span>Green Customer Insights</span></div></div>')


def hero(title: str, description: str, *, label: str, customer: bool = False, emphasis: str | None = None):
    heading = escape(title)
    if emphasis and title.endswith(emphasis):
        heading = escape(title[:-len(emphasis)].rstrip()) + '&nbsp;<strong>' + escape(emphasis) + '</strong>'
    html(f'<section class="hero {"customer-hero" if customer else ""}"><div class="hero-copy"><span class="hero-label">{escape(label)}</span><h1>{heading}</h1><p>{escape(description)}</p></div><img src="{asset("neighbourhood.svg")}" alt="Illustrated neighbourhood with solar roofs, an electric car and trees" /></section>')


def staff_banner():
    html('<div class="staff-banner"><span class="staff-pill">Staff preview</span><p>This page shows what the selected customer would see in their banking app. '
         'Items marked <strong>staff only</strong> are for bank teams and are never shown to the customer.</p></div>')


def staff_tag(text: str):
    html(f'<div class="staff-tag">{escape(text)}</div>')


def section(title: str, description: str = ''):
    html(f'<div class="section-title"><h2>{escape(title)}</h2>{f"<p>{escape(description)}</p>" if description else ""}</div>')


def product_header(code: str, title: str, headline: str, rank: int):
    spec = PRODUCTS[code]
    html(f'<div class="product-header"><div><span class="product-tag">{escape(spec["tag"])}</span><h2>{escape(title)}</h2><p>{escape(headline)}</p><span class="quiet-badge">Priority {rank}</span></div><img src="{asset(spec["asset"])}" alt="{escape(spec["short"])} illustration" /></div>')


def impact_tile(value, unit: str, title: str):
    formatted = ('Unavailable' if value is None else (f'{value:,.0f}' if unit == 'sheets/year' else f'{value:,.3f}' if 0 < value < 1 else f'{value:,.1f}'))
    html(f'<div class="impact-tile"><span>{escape(title)}</span><strong>{formatted}<small>{escape(unit) if value is not None else ""}</small></strong><em>Estimated annual potential</em></div>')


def empty_state(title: str, text: str):
    html(f'<div class="empty-state"><img src="{asset("paperless.svg")}" alt="" /><h3>{escape(title)}</h3><p>{escape(text)}</p></div>')


def feedback_notice(text: str):
    st.success(text, icon=':material/check_circle:')


def friendly_code(code: str) -> str:
    labels = {
        'ALREADY_HAS_EV': 'An EV is already recorded; no replacement use case is available.',
        'VEHICLE_FUEL_CONFLICT': 'Vehicle and fuel records disagree. Confirm the current vehicle first.',
        'VEHICLE_SOLD': 'The vehicle was sold. Old fuel spending is not a current replacement baseline.',
        'NO_EV_PURCHASE_USE_CASE': 'No supported EV purchase use case is recorded.',
        'NOT_HOMEOWNER': 'This product requires home ownership.',
        'ALREADY_HAS_SOLAR': 'Solar is already installed.',
        'ALREADY_DIGITAL': 'Statements are already digital.',
        'NO_MARKETING_CONSENT': 'Finance opportunities are hidden because marketing consent is absent.',
        'MISSING_VEHICLE_TYPE': 'The vehicle type is missing.',
        'PROPERTY_REQUIRES_SUITABILITY_REVIEW': 'Property suitability needs review before a solar opportunity can be shown.',
        'MISSING_HOME_UPGRADE_EVIDENCE': 'There is not enough evidence of a supported home upgrade.',
        'PAPER_STATEMENTS_REQUIRED': 'Paper statements are needed for this customer.',
    }
    return labels.get(code, code.replace('_', ' ').capitalize())


def product_states(states):
    parts = []
    for code, (state, description) in states.items():
        spec = PRODUCTS[code]
        parts.append(f'<div class="product-state {escape(state)}" data-state="{escape(state)}"><img src="{asset(spec["asset"])}" alt="" /><div><strong>{escape(spec["short"])}</strong><span>{escape(description)}</span></div></div>')
    html('<div class="product-states">' + ''.join(parts) + '</div>')


def plain_markdown(text: str) -> str:
    """Display generated prose literally: no remote images/links, HTML or math."""
    return re.sub(r'([\\`*_{}\[\]()#+!|<>~$-])', r'\\\1', text)


def landing_header():
    html('<header class="landing-header"><div class="landing-brand"><div class="brand"><span class="brand-mark" aria-hidden="true">v</span><div>verdant<span>Green Customer Insights</span></div></div></div><span class="landing-note">For relationship, product and sustainability teams</span></header>')


def landing_copy():
    html('<section class="landing-copy"><span class="hero-label">Green banking, with perspective</span><h1><span class="motion-line"><span class="motion-text">Small steps.</span></span><span class="motion-line"><strong class="motion-text">A greener future.</strong></span></h1><p>See which customers a green product genuinely fits, understand the potential impact, and preview exactly what each customer would be shown.</p></section>')


def landing_art():
    # Layer the existing authored vector without changing its geometry/content.
    # Streamlit's HTML sanitizer strips inline SVG: use local data-URI layers.
    scene = (ASSETS / 'neighbourhood.svg').read_text()
    layers = []
    for name, cx in [('halo', '475'), ('sun', '618')]:
        circle = re.search(rf'<circle cx="{cx}"[^>]*/>', scene).group(0)
        layers.append((name, '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 360">' + circle + '</svg>'))
        scene = scene.replace(circle, '')
    layers.append(('neighbourhood', scene))
    images = ''.join(f'<img class="scene-{name}" src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}" alt="" />' for name, svg in layers)
    html(f'<div class="landing-art"><div class="landing-scene" role="img" aria-label="A neighbourhood with solar roofs, an electric car and trees"><div class="scene-depth">{images}</div></div><div class="landing-products"><span>EV finance</span><span>Green home</span><span>Solar</span><span>Paperless</span></div></div>')


def landing_principles():
    html('<section class="landing-principles" aria-label="What to expect"><div><h2>Relevant to each customer.</h2><p>Opportunities come from each customer’s circumstances and banking behaviour, filtered by clear product rules.</p></div><div><h2>Clear about the impact.</h2><p>See potential annual savings, the evidence and the assumptions behind each estimate.</p></div><div><h2>Always the customer’s choice.</h2><p>Scores are review prompts, never decisions. Customer feedback shapes what they are shown next.</p></div></section>')


def intro_splash():
    """A CSS-only arrival sequence; it ends on its own, and JS only lets people skip it."""
    word = ''.join(f'<span style="--i:{i}">{escape(ch)}</span>' for i, ch in enumerate('verdant'))
    icons = ''.join(f'<img style="--i:{i}" src="{asset(spec["asset"])}" alt="" />' for i, spec in enumerate(PRODUCTS.values()))
    petals = ''.join(f'<span class="intro-petal" style="--x:{x}%;--d:{d}ms;--r:{r}deg"></span>'
                     for x, d, r in [(12, 200, -30), (27, 900, 40), (71, 450, -60), (86, 1200, 20), (55, 1600, -15), (40, 300, 70)])
    html('<div class="intro-splash" aria-hidden="true"><div class="intro-glow"></div>' + petals +
         '<div class="intro-stage"><div class="intro-sprout"><span class="intro-sun"></span><span class="intro-soil"></span>'
         '<span class="intro-stem"></span><span class="intro-leaf left"></span><span class="intro-leaf right"></span><span class="intro-leaf top"></span></div>'
         f'<div class="intro-word">{word}</div><p class="intro-tagline">Small steps. A greener future.</p>'
         f'<div class="intro-products">{icons}</div></div><span class="intro-skip">Click or press Esc to skip</span></div>')
