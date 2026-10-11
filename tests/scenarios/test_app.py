from pathlib import Path
from streamlit.testing.v1 import AppTest
import streamlit as st
import pytest
from src.recommend.demo import demo_bundle
from src.recommend.service import RecommendationService

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    monkeypatch.setenv('GREEN_MEMORY_PATH', ':memory:')
    monkeypatch.setenv('LLM_PROVIDER', 'offline')
    from src.recommend.demo import write_demo
    path = ROOT / 'data/processed/test_demo_bundle.json'
    monkeypatch.setenv('GREEN_BUNDLE_PATH', str(path))
    write_demo(path)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()
    path.unlink(missing_ok=True)


def workspace_app():
    app = AppTest.from_file(str(ROOT / 'app/main.py')).run(timeout=20)
    app.button('landing-business').click().run(timeout=20)
    assert not app.exception
    return app


def open_customer(app):
    app.sidebar.radio[0].set_value('Customer view (preview)').run(timeout=20)
    assert not app.exception
    return app


def select_product(app, code):
    next(r for r in app.radio if r.label == 'Choose an opportunity').set_value(code).run(timeout=20)
    assert not app.exception


def test_dashboard_customer_and_action():
    app = workspace_app()
    assert not app.exception
    assert len(app.metric) == 9  # 7 portfolio metrics + 2 optional-product counts
    open_customer(app)
    select_product(app, 'GP01')
    app.button('action-C10001-GP01').click().run(timeout=20)
    assert not app.exception
    assert any('does not submit an application' in x.value for x in app.info)
    assert any(x.proto.label.startswith('Download review brief') for x in app.get('download_button'))
    app.selectbox('customer_id').set_value('C10005').run(timeout=20)
    assert not app.exception
    assert not any((b.key or '').startswith('action-') for b in app.button)
    assert app.button('clear-feedback').disabled


def test_dashboard_displays_p4_headline_and_explanation(monkeypatch):
    backend = RecommendationService(demo_bundle(), insight_provider=lambda row:
        {'headline': 'P4 custom headline', 'text': 'P4 custom explanation'})
    from src.memory.store import MemoryStore
    backend.memory_store = MemoryStore(':memory:')
    # The UI entrypoint imports this function anew on each test run.
    monkeypatch.setattr('src.recommend.runtime.load_runtime_service', lambda path=None: backend)
    st.cache_resource.clear()
    app = workspace_app()
    open_customer(app)
    displayed = [item.value for item in app.markdown]
    assert 'P4 custom headline' in displayed
    assert 'P4 custom explanation' in displayed
    assert not any('You may want to explore' in value for value in displayed)


def test_segment_button_navigates_directly_to_customer():
    app = workspace_app()
    expected = app.selectbox('segment-customer').value
    app.button('open-customer').click().run(timeout=20)
    assert not app.exception
    assert app.sidebar.radio[0].value == 'Customer view (preview)'
    assert app.selectbox('customer_id').value == expected


def test_feedback_hides_then_erasure_restores_scores():
    app = open_customer(workspace_app())
    select_product(app, 'GP01')
    original_metrics = [m.value for m in app.metric]
    app.button('dismissed-C10001-GP01').click().run(timeout=20)
    assert not app.exception
    assert any('hidden by your feedback' in x.value for x in app.info)
    assert not any(b.key == 'action-C10001-GP01' for b in app.button)
    assert any('hidden for 30 days' in x.value for x in app.success)
    app.button('clear-feedback').click().run(timeout=20)
    app.button('confirm-clear-feedback').click().run(timeout=20)
    select_product(app, 'GP01')
    assert [m.value for m in app.metric] == original_metrics
    assert not any('hidden by your feedback' in x.value for x in app.info)


def test_provider_markup_is_never_trusted_html(monkeypatch):
    from src.memory.store import MemoryStore
    malicious = '<img src=x onerror=alert(1)>'
    backend = RecommendationService(demo_bundle(), insight_provider=lambda row:
        {'headline': malicious, 'text': malicious}, memory_store=MemoryStore(':memory:'))
    monkeypatch.setattr('src.recommend.runtime.load_runtime_service', lambda path=None: backend)
    st.cache_resource.clear()
    app = open_customer(workspace_app())
    from app.ui import plain_markdown
    rendered = [block for block in app.markdown if block.value == plain_markdown(malicious)]
    assert rendered
    assert all(not block.proto.allow_html for block in rendered)


def test_segment_filters_handle_empty_selection():
    app = workspace_app()
    app.selectbox('region-filter').set_value('London').run(timeout=20)
    app.selectbox('product-filter').set_value('GP01').run(timeout=20)
    assert not app.exception
    # Choosing a product/region never alters the global cohort metrics.
    assert len(app.metric) == 9  # 7 portfolio metrics + 2 optional-product counts


def test_customer_and_region_filters_survive_workspace_navigation():
    app = workspace_app()
    region = app.selectbox('region-filter').options[-1]
    app.selectbox('region-filter').set_value(region).run(timeout=20)
    app.selectbox('product-filter').set_value('GP02').run(timeout=20)
    open_customer(app)
    app.selectbox('customer_id').set_value('C10002').run(timeout=20)
    app.button('back-to-portfolio').click().run(timeout=20)
    assert app.selectbox('region-filter').value == region
    assert app.selectbox('product-filter').value == 'GP02'
    open_customer(app)
    assert app.selectbox('customer_id').value == 'C10002'


def test_not_relevant_has_explicit_180_day_cooloff_and_clear_can_cancel():
    app = open_customer(workspace_app())
    select_product(app, 'GP01')
    app.button('not_relevant-C10001-GP01').click().run(timeout=20)
    assert any('hidden for 180 days' in x.value for x in app.success)
    app.button('clear-feedback').click().run(timeout=20)
    assert app.button('confirm-clear-feedback')
    app.button('cancel-clear-feedback').click().run(timeout=20)
    assert any('hidden by your feedback' in x.value for x in app.info)


def test_generated_markdown_images_and_currency_are_literal(monkeypatch):
    from src.memory.store import MemoryStore
    from app.ui import plain_markdown
    text = '![tracking](https://example.invalid/track?id=C10001) costs $1 to $2'
    backend = RecommendationService(demo_bundle(), insight_provider=lambda row:
        {'headline': 'Plain headline', 'text': text}, memory_store=MemoryStore(':memory:'))
    monkeypatch.setattr('src.recommend.runtime.load_runtime_service', lambda path=None: backend)
    st.cache_resource.clear()
    app = open_customer(workspace_app())
    rendered = [m for m in app.markdown if m.value == plain_markdown(text)]
    assert len(rendered) == 1
    assert not rendered[0].proto.allow_html
    assert r'\!\[tracking\]' in rendered[0].value
    assert r'\$1' in rendered[0].value


def test_selected_insight_only_calls_provider_for_visible_requested_product():
    called = []
    def provider(row):
        called.append(row['product_code'])
        return {'headline': 'Headline', 'text': 'Explanation'}
    backend = RecommendationService(demo_bundle(), insight_provider=provider)
    result = backend.insight('C10001', product_code='GP01')
    assert called == ['GP01']
    assert [i['product_code'] for i in result['insights']] == ['GP01']
    called.clear()
    assert backend.insight('C10005', product_code='GP01')['insights'] == []
    assert not called
    with pytest.raises(ValueError, match='Unknown product'):
        backend.insight('C10001', product_code='invalid')


def test_stylesheet_rules_never_enter_page_prose():
    app = workspace_app()
    assert not app.exception
    # Markdown parsed the old stylesheet as visible text after blank lines.
    prose = '\n'.join(block.value for block in app.markdown)
    assert '.product-states {' not in prose
    assert 'grid-template-columns:repeat' not in prose
    assert '@font-face' not in prose
    open_customer(app)
    assert '.product-states {' not in '\n'.join(block.value for block in app.markdown)


def test_landing_entry_points_and_return_home():
    app = AppTest.from_file(str(ROOT / 'app/main.py')).run(timeout=20)
    assert not app.exception
    assert not app.metric
    assert not app.sidebar.radio
    assert app.button('landing-customer').label == 'Preview a customer view'
    app.button('landing-customer').click().run(timeout=20)
    assert not app.exception
    assert app.sidebar.radio[0].value == 'Customer view (preview)'
    assert app.query_params['page'] == ['customer']
    app.sidebar.radio[0].set_value('Home').run(timeout=20)
    assert not app.exception
    assert app.query_params['page'] == ['home']
    assert not app.metric
    app.button('landing-business').click().run(timeout=20)
    assert len(app.metric) == 9  # 7 portfolio metrics + 2 optional-product counts
    assert app.query_params['page'] == ['business']


def test_workspace_deep_links():
    for slug, label in [('business', 'Business dashboard'), ('customer', 'Customer view (preview)')]:
        app = AppTest.from_file(str(ROOT / 'app/main.py'))
        app.query_params['page'] = slug
        app.run(timeout=20)
        assert not app.exception
        assert app.sidebar.radio[0].value == label


def test_landing_does_not_require_customer_bundle(monkeypatch):
    monkeypatch.setenv('GREEN_BUNDLE_PATH', '/tmp/verdant-nonexistent-bundle.json')
    app = AppTest.from_file(str(ROOT / 'app/main.py')).run(timeout=20)
    assert not app.exception
    assert not app.error
    assert app.button('landing-business')
    app.button('landing-business').click().run(timeout=20)
    assert not app.exception
    assert any('Customer data could not be loaded' in x.value for x in app.error)


def test_optional_products_card_on_preview(monkeypatch):
    """Green Loans appears for C10013 as a rule-based signal with no likelihood shown."""
    monkeypatch.setenv('GREEN_BUNDLE_PATH', str(ROOT / 'data/preview/team_bundle.json'))
    st.cache_resource.clear()
    app = workspace_app()
    assert any(m.label == 'Green Loans' for m in app.metric)
    open_customer(app)
    app.selectbox('customer_id').set_value('C10013').run(timeout=20)
    assert not app.exception
    assert any('Rule-based signal' in c.value for c in app.caption)
    assert any('mortgage' in m.value for m in app.markdown) or any('mortgage' in str(t.value) for t in app.get('text'))
    app.button('optional-action-C10013-GP06').click().run(timeout=20)
    assert not app.exception
    assert any('Nothing has been applied for' in x.value for x in app.info)
