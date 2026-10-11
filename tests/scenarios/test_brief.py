from app.brief import brief_content, review_pdf
from src.recommend.demo import demo_bundle
from src.recommend.service import RecommendationService

STEPS = ['Arrange an assessment.', 'Compare terms with your relationship team.']


def rows():
    backend = RecommendationService(demo_bundle())
    return [r for cid in backend.customers for r in backend.recommendations(cid)['recommendations']]


def test_review_brief_is_a_pdf_for_every_shown_opportunity():
    for row in rows():
        data = review_pdf(row, {'headline': 'A greener step', 'text': 'Plain explanation.'}, STEPS)
        assert data.startswith(b'%PDF') and data.rstrip().endswith(b'%%EOF')


def test_brief_content_is_plain_language():
    row = {**rows()[0], 'impact': {'status': 'ESTIMATED', 'kg_co2e_per_year': 119.859, 'energy_kwh_per_year': 832.6,
                                   'paper_sheets_per_year': None, 'method': 'Illustrative method',
                                   'assumptions': ['One assumption.'], 'horizon_years': 1}}
    content = brief_content(row, None, STEPS)
    assert content['reasons'] and all(r[0].isupper() and r.endswith('.') for r in content['reasons'])
    assert all('_' not in label for label, _ in content['evidence'])
    assert content['impact'] == [('120', 'kg CO₂e', 'emissions avoided'), ('833', 'kWh', 'less energy bought')]
    assert content['likelihood'].endswith('%')
    assert review_pdf(row, None, STEPS).startswith(b'%PDF')
    assert content['steps'] == STEPS
    assert 'not a credit approval' in ' '.join(content['notes'])
