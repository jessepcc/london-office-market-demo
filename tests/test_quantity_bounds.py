import json
from pathlib import Path

import pytest

from london_market.claim_audit import quantity_bound_issues
from london_market.domain import Claim, Evidence


def source(text, **updates):
    packet=json.loads((Path(__file__).parent/'fixtures/completion/attempt9-completion-source-packet.json').read_text())
    item=Evidence.model_validate(packet['evidence'][0])
    return item.model_copy(update={'excerpt':text, **updates})


@pytest.mark.parametrize('kind', ['fact','forecast','interpretation','watchpoint'])
def test_actual_laboratory_ceiling_applies_to_every_claim_kind(kind):
    packet=json.loads((Path(__file__).parent/'fixtures/completion/attempt9-completion-source-packet.json').read_text())
    raw={**packet['candidate']['claims'][33],'kind':kind}
    c=Claim.model_validate(raw)
    sources=[Evidence.model_validate(e) for e in packet['evidence'] if e['id'] in c.evidence_ids]
    issues=quantity_bound_issues(c,sources)
    assert any('up to 60 per cent' in issue and '60%' in issue for issue in issues)
    assert quantity_bound_issues(c.model_copy(update={'text':c.text.replace('the 60% laboratory allocation','the original allocation of up to 60% for laboratories')}), sources)==[]


@pytest.mark.parametrize('phrase,faithful,wrong', [('up to 60% for laboratories','at most 60% for laboratories','60% for laboratories'), ('over 60% occupancy','more than 60% occupancy','at least 60% occupancy'), ('at least 60% occupancy','no less than 60% occupancy','over 60% occupancy'), ('approximately £1 million refurbishment cost','about £1,000,000 refurbishment cost','£1,000,000 refurbishment cost'), ('under 300k sq ft of laboratories','less than 300,000 square feet of laboratories','up to 300,000 square feet of laboratories')])
def test_normalized_units_operators_and_strictness(phrase,faithful,wrong):
    e=source(phrase)
    assert quantity_bound_issues(Claim(kind='fact',text=faithful,evidence_ids=[e.id]),[e])==[]
    assert quantity_bound_issues(Claim(kind='fact',text=wrong,evidence_ids=[e.id]),[e])


def test_unrelated_equal_quantities_units_and_exact_sources_do_not_conflate():
    bounded=source('Up to 60% of laboratories could be occupied.')
    exact=source('Offices have 60% occupancy; rents are £60 per sq ft.', id='exact')
    assert quantity_bound_issues(Claim(kind='fact',text='Offices have 60% occupancy; rents are £60 per sq ft.',evidence_ids=[bounded.id,exact.id]),[bounded,exact])==[]
    unrelated=source('Laboratories have 60% occupancy.',id='lab-exact')
    c=Claim(kind='fact',text='Laboratories have 60% occupancy.',evidence_ids=[bounded.id,unrelated.id])
    assert quantity_bound_issues(c,[bounded,unrelated])==[]
    different=unrelated.model_copy(update={'native_geography':'New York','mapped_geography':'Unmapped'})
    assert quantity_bound_issues(c,[bounded,different])


def test_headline_bound_is_checked_separately_and_ambiguity_is_explicit():
    e=source('Up to 60% of space is earmarked for laboratories.')
    c=Claim(kind='watchpoint',headline='60% laboratory allocation',text='Track the original plan for up to 60% of laboratories.',evidence_ids=[e.id])
    assert any('headline' in issue for issue in quantity_bound_issues(c,[e]))
    assert any('ambiguous' in issue for issue in quantity_bound_issues(c.model_copy(update={'headline':'The allocation is 60%.'}),[e]))


def test_explicit_periods_and_adjacent_quantities_cannot_supply_another_occurrence():
    a=source('Up to 20% of City office space was available in Q1 2025.')
    b=source('20% of City office space was available in Q2 2026.',id='new-period')
    c=Claim(kind='fact',text='20% of City office space was available in Q1 2025.',evidence_ids=[a.id,b.id])
    assert quantity_bound_issues(c,[a,b])
    assert quantity_bound_issues(c.model_copy(update={'text':'20% of City office space was available in Q2 2026.'}),[a,b])==[]
    e=source('Up to 60% for laboratories, and 60% occupancy for offices.')
    assert quantity_bound_issues(Claim(kind='fact',text='60% occupancy for offices.',evidence_ids=[e.id]),[e])==[]
    assert quantity_bound_issues(Claim(kind='fact',text='60% for laboratories.',evidence_ids=[e.id]),[e])


def test_currency_per_square_foot_equivalent_unit_keeps_approximation():
    e=source('About £60 psf achieved rent for laboratories.')
    for text in ('Approximately £60 per square foot achieved rent for laboratories.', 'About 60 GBP per square foot achieved rent for laboratories.'):
        assert quantity_bound_issues(Claim(kind='fact',text=text,evidence_ids=[e.id]),[e])==[]


@pytest.mark.parametrize('index',[4,7])
def test_actual_faithfully_preserved_reuters_and_occupier_bounds_are_not_rejected(index):
    p=json.loads((Path(__file__).parent/'fixtures/completion/attempt9-completion-source-packet.json').read_text())
    c=Claim.model_validate(p['candidate']['claims'][index])
    assert quantity_bound_issues(c,[Evidence.model_validate(e) for e in p['evidence'] if e['id'] in c.evidence_ids])==[]


def test_normalized_source_units_do_not_create_numeric_false_positives():
    from london_market.claim_audit import unsupported_figures
    e=source('Approximately £1 million refurbishment cost; under 300k sq ft of laboratories.')
    c=Claim(kind='fact',text='About £1,000,000 refurbishment cost and less than 300,000 square feet of laboratories.',evidence_ids=[e.id])
    assert unsupported_figures(c,[e])==[]
    assert quantity_bound_issues(c,[e])==[]
    assert unsupported_figures(c.model_copy(update={'text':'About £2,000,000 refurbishment cost.'}),[e])==[2000000]
