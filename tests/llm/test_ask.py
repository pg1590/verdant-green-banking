"""Ask the portfolio: the LLM routes and phrases; the service computes; invented numbers never pass."""
import json
from pathlib import Path

import pytest

from src.llm.ask import ANSWER_SYSTEM, ROUTER_SYSTEM, check_answer, facts_for, keyword_route
from src.llm.client import LLMUnavailable, OfflineProvider
from src.recommend.service import RecommendationService

PREVIEW = Path(__file__).resolve().parents[2] / 'data' / 'preview' / 'team_bundle.json'


@pytest.fixture(scope='module')
def service():
    return RecommendationService(json.loads(PREVIEW.read_text(encoding='utf-8')))


class FakeProvider:
    model = 'fake'

    def __init__(self, route, answers):
        self.route, self.answers = route, list(answers)

    def complete_json(self, system, user, schema):
        if system == ROUTER_SYSTEM:
            return self.route
        assert system == ANSWER_SYSTEM and 'FACTS' in user
        return {'answer': self.answers.pop(0)}


def test_keyword_router_offline():
    assert keyword_route('How many solar opportunities in London?') == {'intent': 'region', 'product_code': 'GP03', 'region': 'London'}
    assert keyword_route('What is the total CO2 impact?')['intent'] == 'impact'
    assert keyword_route('Tell me about green investments')['intent'] == 'optional_products'
    assert keyword_route('What is the weather?')['intent'] == 'out_of_scope'


def test_grounded_llm_answer_is_used(service):
    facts = facts_for(service, {'intent': 'product', 'product_code': 'GP04', 'region': None})
    shown = facts['facts'][0].split(' customers')[0]
    route = {'intent': 'product', 'product_code': 'GP04', 'region': 'none'}
    out = service.ask('How is paperless doing?', FakeProvider(route, [f'{shown} customers are shown Paperless Banking.']))
    assert out['provenance']['text_source'] == 'LLM' and out['route']['region'] is None


def test_invented_numbers_and_approval_language_fall_back_to_facts(service):
    route = {'intent': 'impact', 'product_code': 'none', 'region': 'none'}
    out = service.ask('Total CO2?', FakeProvider(route, ['We will save 999,999 tonnes.', 'Customers are approved, saving 123 tonnes.']))
    assert out['provenance']['text_source'] == 'TEMPLATE'
    assert any('numbers not in the facts' in p for p in out['provenance']['rejected'])
    assert out['answer'] == ' '.join(out['facts']['facts'])
    assert check_answer('Customers are pre-approved.', {'facts': ['x']}) == ['banned language: eligibility_claim']


def test_out_of_scope_and_offline(service):
    out = service.ask('Is C10001 approved?', FakeProvider({'intent': 'out_of_scope', 'product_code': 'none', 'region': 'none'}, []))
    assert out['route']['intent'] == 'out_of_scope' and 'cannot discuss individual customers' in out['answer']
    offline = service.ask('How many paperless opportunities?', OfflineProvider())
    assert offline['provenance']['router'] == 'KEYWORDS' and offline['route']['product_code'] == 'GP04'
    with pytest.raises(ValueError):
        service.ask('   ', OfflineProvider())


def test_provider_failure_still_answers(service):
    class Broken:
        model = 'broken'
        def complete_json(self, *a):
            raise LLMUnavailable('down')
    out = service.ask('Where should we focus?', Broken())
    assert out['provenance']['llm_error'] == 'down' and out['route']['intent'] == 'top_segments' and out['answer']
