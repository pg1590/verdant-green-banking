from copy import deepcopy
import pytest
from src.recommend.demo import demo_bundle
from src.recommend.engine import build_recommendations
from src.recommend.service import RecommendationService


def inputs():
    bundle = demo_bundle()
    return bundle['customers'][0], bundle['predictions']['C10001']


def test_eligibility_before_ranking():
    customer, predictions = inputs()
    customer['home_status'] = 'renter'
    predictions['GP03']['propensity'] = 1
    predictions['GP03']['confidence'] = 1
    rows = build_recommendations(customer, predictions)
    solar = next(r for r in rows if r['product_code'] == 'GP03')
    assert solar['ranking_score'] is None and solar['rank'] is None
    assert solar['action'] is None and not solar['eligible_to_show']
    assert rows[0]['product_code'] == 'GP04'


def test_completeness_cap_does_not_penalise_twice():
    customer, predictions = inputs()
    customer['data_completeness'] = .5
    predictions['GP01']['confidence'] = .4
    row = next(r for r in build_recommendations(customer, predictions) if r['product_code'] == 'GP01')
    assert row['model_confidence'] == .4 and row['confidence'] == .4
    assert row['ranking_score'] == pytest.approx(.84 * .4)


@pytest.mark.parametrize('field,value', [('propensity',float('nan')),('confidence',1.1),('confidence',True),('model_version',None),('reason_codes',[]),('evidence',{})])
def test_invalid_prediction_suppressed(field,value):
    customer,predictions = inputs()
    predictions['GP01'][field] = value
    row = next(r for r in build_recommendations(customer,predictions) if r['product_code'] == 'GP01')
    assert not row['eligible_to_show'] and row['action'] is None


def test_no_impact_does_not_invent_emissions():
    customer,predictions = inputs()
    rows = build_recommendations(customer,predictions)
    assert rows[0]['impact']['status'] == 'INSUFFICIENT_DATA'
    assert rows[0]['impact']['kg_co2e_per_year'] is None
    assert rows[0]['eligible_to_show']


def test_sourced_impact_and_conservative_portfolio_sum():
    bundle = demo_bundle()
    bundle['customers'] = bundle['customers'][:1]
    def impact(n):
        return {'kg_co2e_per_year': n, 'status': 'ESTIMATED', 'method': 'test fixture method',
                'assumptions': ['test fixture only'], 'horizon_years':1, 'factor_edition': 'test-edition-v1'}
    bundle['impacts']['C10001'] = {'GP01':impact(100), 'GP02':impact(200), 'GP03':impact(300), 'GP04':impact(10)}
    summary = RecommendationService(bundle).summary()
    assert summary['potential_impact']['kg_co2e_per_year']['value'] == 100
    assert summary['impact_coverage_customers'] == 1


def test_missing_method_or_bad_impact_is_unavailable():
    customer,predictions = inputs()
    rows = build_recommendations(customer,predictions, {'GP04': {'status':'ESTIMATED', 'kg_co2e_per_year':123}})
    assert rows[0]['impact']['status'] == 'INSUFFICIENT_DATA'


def test_strict_bundle_and_no_silent_team_mock_mix():
    bundle = demo_bundle()
    bundle['data_mode'] = 'TEAM_OUTPUTS'
    with pytest.raises(ValueError, match='Mock predictions'):
        RecommendationService(bundle)
    bundle = demo_bundle()
    bundle['customers'].append(deepcopy(bundle['customers'][0]))
    with pytest.raises(ValueError, match='unique'):
        RecommendationService(bundle)


def test_insights_only_for_eligible_and_provider_only_gets_structured_rows():
    bundle = demo_bundle()
    seen = []
    def provider(row):
        seen.append(row)
        return 'Validated test fixture'
    service = RecommendationService(bundle, insight_provider=provider)
    result = service.insight('C10002')
    assert {r['product_code'] for r in result['insights']} == {'GP01','GP04'}
    assert all(r['eligible_to_show'] for r in seen)
    assert all(r['text'] == 'Validated test fixture' and r['source'] == 'P4_VALIDATED_PROVIDER'
               and r['fallback_reason'] is None for r in result['insights'])
    assert service.summary()['potential_impact']['kg_co2e_per_year']['value'] is None
    with pytest.raises(KeyError):
        service.recommendations('unknown')


def test_structured_p4_explanation_is_preserved_without_template_fallback():
    seen = []
    def provider(row):
        seen.append(row['product_code'])
        return {'headline': f"P4 headline {row['product_code']}", 'text': 'P4 validated explanation'}
    service = RecommendationService(demo_bundle(), insight_provider=provider)
    result = service.insight('C10002')
    assert set(seen) == {'GP01', 'GP04'}
    for item in result['insights']:
        assert item['headline'] == f"P4 headline {item['product_code']}"
        assert item['text'] == 'P4 validated explanation'
        assert item['source'] == 'P4_VALIDATED_PROVIDER'
        assert item['fallback_reason'] is None


@pytest.mark.parametrize('output', [None, {}, {'headline': 'Title'},
    {'headline': ' ', 'text': 'Body'}, {'headline': 'Title', 'text': ''},
    {'headline': 'Title', 'text': 42}, ['Title', 'Body']])
def test_invalid_p4_output_has_explicit_fallback_reason(output):
    service = RecommendationService(demo_bundle(), insight_provider=lambda row: output)
    for item in service.insight('C10002')['insights']:
        assert item['source'] == 'DETERMINISTIC_TEMPLATE'
        assert item['fallback_reason'] == 'INVALID_PROVIDER_OUTPUT'
        assert item['headline'] == item['recommendation']['product_name']


def test_missing_p4_provider_has_explicit_fallback_reason():
    assert all(item['fallback_reason'] == 'MISSING_PROVIDER'
               for item in RecommendationService.demo().insight('C10002')['insights'])


def test_missing_timestamp_and_zero_completeness_fail_closed():
    customer,predictions = inputs()
    customer['feature_cutoff_date'] = '2026-13-30'
    assert all(not r['eligible_to_show'] for r in build_recommendations(customer,predictions))
    customer['feature_cutoff_date'] = '2026-06-30'
    customer['data_completeness'] = 0
    assert all(not r['eligible_to_show'] for r in build_recommendations(customer,predictions))


@pytest.mark.parametrize('history', [[{'date':'bad','customers_with_opportunities':1}], [{'date':'2026-06-30','customers_with_opportunities':-1}], [{'date':'2026-06-30','customers_with_opportunities':True}], None])
def test_invalid_history_fails_loading(history):
    bundle = demo_bundle()
    bundle['portfolio_history'] = history
    with pytest.raises(ValueError):
        RecommendationService(bundle)


def team_bundle():
    bundle = demo_bundle()
    bundle['data_mode'] = 'TEAM_OUTPUTS'
    for products in bundle['predictions'].values():
        for pred in products.values():
            pred['model_version'] = 'test_model_v1'
            pred.pop('data_mode', None)
    return bundle


def sourced_impact(n, edition='DESNZ GHG Conversion Factors 2026', sheets=None):
    return {'status': 'ESTIMATED', 'kg_co2e_per_year': n,
            'paper_sheets_per_year': sheets, 'factor_edition': edition,
            'method': 'Test fixture only', 'assumptions': ['Test fixture only'], 'horizon_years': 1}


def test_structural_eligibility_does_not_include_marketing_consent():
    service = RecommendationService.demo()
    rows = service.recommendations('C10008', include_suppressed=True)['recommendations']
    for row in rows:
        if row['product_code'] != 'GP04':
            assert row['structurally_eligible'] and not row['eligible_to_show']
            assert 'NO_MARKETING_CONSENT' in row['suppression_reasons']


def test_factor_edition_preserved_and_required_for_estimate():
    customer,predictions = inputs()
    impact = sourced_impact(100)
    row = next(r for r in build_recommendations(customer,predictions, {'GP01':impact}) if r['product_code']=='GP01')
    assert row['impact']['factor_edition'] == 'DESNZ GHG Conversion Factors 2026'
    impact.pop('factor_edition')
    row = next(r for r in build_recommendations(customer,predictions, {'GP01':impact}) if r['product_code']=='GP01')
    assert row['impact']['status'] == 'INSUFFICIENT_DATA'
    assert row['impact']['factor_edition'] is None


def test_mixed_emissions_editions_fail_loading():
    bundle = team_bundle()
    bundle['impacts']['C10001'] = {'GP01':sourced_impact(100,'edition-a'), 'GP03':sourced_impact(100,'edition-b')}
    with pytest.raises(ValueError, match='Mixed emissions factor editions'):
        RecommendationService(bundle)


def test_digital_impact_separate_and_no_missing_finance_fallback():
    bundle = team_bundle()
    bundle['customers'] = bundle['customers'][:1]
    bundle['impacts']['C10001'] = {'GP03':sourced_impact(300), 'GP04':sourced_impact(10,sheets=24)}
    summary = RecommendationService(bundle).summary()
    # GP01 ranks highest among finance: no estimate exists, so GP03 cannot fill it.
    assert summary['potential_impact']['kg_co2e_per_year']['value'] is None
    assert summary['potential_impact']['paper_sheets_per_year']['value'] == 24
    assert summary['digital_impact']['kg_co2e_per_year']['value'] == 10
    assert summary['customers_with_opportunities'] == 1
    assert summary['factor_editions'] == ['DESNZ GHG Conversion Factors 2026']


@pytest.mark.parametrize('changes,message', [
    ({'region':'Yorkshire'}, 'invalid canonical region'),
    ({'income_band':'middle'}, 'invalid canonical income_band'),
    ({'data_completeness':None}, 'requires data_completeness'),
    ({'feature_cutoff_date':None}, 'requires ISO feature_cutoff_date'),
    ({'data_as_of':'2026-07-31'}, 'must match feature_cutoff_date'),
    ({'conflict_flags':['VEHICLE_SLOD']}, 'invalid conflict_flags'),
    ({'data_flags':'UNRECOGNISED_FLAG'}, 'unknown data_flags'),
    ({'took_up':1}, 'labels/hidden traits'),
    ({'eco_mindedness':.8}, 'labels/hidden traits'),
])
def test_team_bundle_contract_validation(changes,message):
    bundle = team_bundle()
    bundle['customers'][0].update(changes)
    with pytest.raises(ValueError, match=message):
        RecommendationService(bundle)


def test_quality_flags_pass_through_without_second_penalty():
    bundle = team_bundle()
    bundle['customers'][0]['data_flags'] = 'RECENT_CIRCUMSTANCE_CHANGE,ENERGY_APPORTIONED_FROM_DUAL_FUEL'
    rows = RecommendationService(bundle).recommendations('C10001')['recommendations']
    row = next(r for r in rows if r['product_code']=='GP01')
    assert row['data_flags'] == bundle['customers'][0]['data_flags']
    assert row['confidence'] == row['model_confidence'] == .87


@pytest.mark.parametrize('field', ['conflict_flags','data_flags'])
def test_team_bundle_requires_explicit_flag_keys(field):
    bundle = team_bundle()
    bundle['customers'][0].pop(field)
    with pytest.raises(ValueError, match='requires ' + field):
        RecommendationService(bundle)


def test_uniform_but_nonconfigured_edition_is_rejected():
    bundle = team_bundle()
    bundle['impacts']['C10001']['GP01'] = sourced_impact(100,edition='wrong-edition')
    with pytest.raises(ValueError, match='must match settings'):
        RecommendationService(bundle)


def test_malformed_team_estimate_has_explicit_integration_error():
    bundle = team_bundle()
    bundle['impacts']['C10001']['GP01'] = sourced_impact(100)
    bundle['impacts']['C10001']['GP01'].pop('factor_edition')
    with pytest.raises(ValueError, match='malformed ESTIMATED impact'):
        RecommendationService(bundle)


def test_energy_only_estimate_is_not_counted_as_co2e_coverage():
    bundle = team_bundle()
    bundle['customers'] = bundle['customers'][:1]
    impact = sourced_impact(None)
    impact['energy_kwh_per_year'] = 100
    bundle['impacts']['C10001']['GP01'] = impact
    summary = RecommendationService(bundle).summary()
    assert summary['finance_estimate_customers'] == 1
    assert summary['impact_coverage_customers'] == 0
    assert summary['potential_impact']['energy_kwh_per_year']['coverage_customers'] == 1
