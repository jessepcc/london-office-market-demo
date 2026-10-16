"""Actual rejected revision and source qualifications stay bounded without truncation."""
import json
import re
from pathlib import Path

import httpx
import pytest

from london_market.domain import Claim, Evidence, Observation, stable_id
from london_market.providers import LANDLORD_LEASING_DEFINITION, OpenRouterProvider, ProviderError, SignalCard, signal_payload
from london_market.runner import ReviseFinishArgs
from london_market.weekly import signal_card


def test_actual_revision_reports_every_conflicting_latest_draft_index():
    raw = json.loads((Path(__file__).parent / 'fixtures/completion/attempt6-invalid-revision.json').read_text())
    for replacement in raw['replacements']:
        if replacement['claim'].get('kind') == 'interpretation':
            replacement['claim'] = {'kind': 'interpretation', 'interpretation_role': 'source_view', 'source_view': {'evidence_id': replacement['claim']['evidence_ids'][0]}}
    with pytest.raises(ValueError, match=r'Conflicting latest draft claim indexes \[38, 40\]; remove each from either drop_claim_indexes OR replacements'):
        ReviseFinishArgs.model_validate(raw)
    raw['drop_claim_indexes'] = [index for index in raw['drop_claim_indexes'] if index not in (38, 40)]
    assert [item.index for item in ReviseFinishArgs.model_validate(raw).replacements][-2:] == [38, 40]


def test_actual_ons_fact_keeps_all_exact_definitions_and_rejects_oversized_rubric_state():
    from london_market.config import Settings
    from london_market.claim_audit import unsupported_figures
    snapshot = json.loads((Path(__file__).parent / 'fixtures/completion/attempt6-ons-signal.json').read_text())
    evidence = {item['id']: Evidence.model_validate(item) for item in snapshot['evidence']}
    observations = [Observation.model_validate(item) for item in snapshot['observations']]
    original = json.dumps(snapshot, sort_keys=True)

    class View:
        def get_evidence(self, eid, mode): return evidence[eid]
        def list_observations(self, mode): return observations

    claim = Claim.model_validate(snapshot['claim'])
    assert unsupported_figures(claim, [evidence[eid] for eid in claim.evidence_ids]) == [16, 64]
    corrected = claim.model_copy(update={'evidence_ids': claim.evidence_ids + [snapshot['age_header_id'], snapshot['residence_definition_id']]})
    assert unsupported_figures(corrected, [evidence[eid] for eid in corrected.evidence_ids]) == []
    card = signal_card(corrected, View(), 'live')
    assert card.text == corrected.headline + '\n' + corrected.text
    assert card.fingerprint == stable_id('card', corrected.kind, corrected.text, corrected.headline, sorted(corrected.evidence_ids), sorted(corrected.chart_ids))
    for item in evidence.values():
        for qualification in re.split(r' Source (?:qualifications|date qualification)\. ', item.validation_reason)[1:]:
            assert qualification in card.qualifications
        for value in (item.native_geography, str(item.published_at), item.validation_status):
            assert value in ' '.join(card.qualifications)
    for observation in observations:
        for value in (observation.metric, observation.native_geography, observation.definition, str(observation.period_start), str(observation.period_end)):
            assert value in ' '.join(card.qualifications)
    assert sum('Residence-based London LFS estimates' in item for item in card.qualifications) == 1
    complete_state = json.dumps({"landlord_leasing_definition": LANDLORD_LEASING_DEFINITION,
                                 "card_0": {"fact": card.text, "qualifications": card.qualifications}},
                                ensure_ascii=False, separators=(",", ":"))
    assert len(complete_state.encode()) == 2324
    requests = []
    def respond(request):
        requests.append(request)
        raise AssertionError("An oversized complete state must fail before HTTP")
    provider = OpenRouterProvider(Settings(openrouter_api_key='synthetic'), transport=httpx.MockTransport(respond))
    with pytest.raises(ProviderError, match="Complete facts and qualifications"):
        provider.classify_signals([card], 1)
    assert requests == []
    assert json.dumps(snapshot, sort_keys=True) == original


def test_signal_boundary_counts_complete_utf8_state_and_never_truncates():
    def card(text): return SignalCard(fingerprint='test', text=text, qualifications=['Distinct material qualification.'])
    overhead = len(signal_payload([card('')], 'clef')['state'][0].encode())
    exact = card('x' * (2000 - overhead))
    assert len(signal_payload([exact], 'clef')['state'][0].encode()) == 2000
    with pytest.raises(ProviderError, match='Complete facts and qualifications'):
        signal_payload([card(exact.text + 'é')], 'clef')
    assert exact.qualifications == ['Distinct material qualification.']


def test_new_ons_contract_does_not_repeat_exact_caveats():
    from test_finalization_followup import parsed
    from london_market.sources import MONITORED_SOURCES
    source = MONITORED_SOURCES['ons-regional-labour-september-2026']
    assert source.version == '4'
    current, _ = parsed(source.id)
    for item in current.evidence:
        for caveat in source.caveats:
            assert item.validation_reason.count(caveat) == 1


def test_retained_definition_pointers_deduplicate_versions_before_the_cap(tmp_path, monkeypatch):
    from dataclasses import replace
    from datetime import timedelta
    from london_market.coverage import retained_coverage_index, reviewed_repair_evidence_ids
    from london_market.sources import MONITORED_SOURCES, parse_monitored_source
    from london_market.store import Store
    from test_finalization_followup import parsed
    source_id = 'ons-regional-labour-september-2026'
    current, page = parsed(source_id)
    with monkeypatch.context() as patch:
        patch.setitem(MONITORED_SOURCES, source_id, replace(MONITORED_SOURCES[source_id], version='3'))
        old = parse_monitored_source(source_id, page)
    newer_page = page.model_copy(update={'document': page.document.model_copy(update={'retrieved_at': page.document.retrieved_at + timedelta(seconds=10)})})
    current = parse_monitored_source(source_id, newer_page)
    with Store(tmp_path / 'db') as store:
        store.ingest(old.document, old.evidence, old.observations)
        store.ingest(current.document, current.evidence, current.observations)
        all_evidence = store.list_evidence('live')
        row = next(item for item in retained_coverage_index({item.id: item for item in all_evidence}, store.list_observations('live')) if item['requirement_id'] == 'employment')
        expected = {item.id for item in current.evidence if item.category == 'labour_definitions'}
        assert len(expected) == 3 and set(row['related_evidence_ids']) == expected
        assert len(reviewed_repair_evidence_ids([current.evidence[0]], all_evidence)) == 6
        assert store.get_evidence(old.evidence[0].id, 'live').validation_reason == old.evidence[0].validation_reason


@pytest.mark.parametrize("option,label", [
    ("improves_landlord_leasing_conditions", "bullish"),
    ("worsens_landlord_leasing_conditions", "bearish"),
    ("no_directional_leasing_effect", "neutral"),
])
def test_semantic_signal_choices_map_exactly_without_changing_values(option, label):
    from london_market.config import Settings
    options = ["improves_landlord_leasing_conditions", "worsens_landlord_leasing_conditions", "no_directional_leasing_effect"]
    probabilities = {key: 0.8 if key == option else 0.1 for key in options}
    card = SignalCard(fingerprint="mock-card", text="A source fact.", qualifications=["Exact source qualification."])
    def respond(request):
        payload = json.loads(request.content)
        assert list(payload["questions"]["card_0"]["criteria"]) == options
        assert list(json.loads(payload["state"][0])) == ["landlord_leasing_definition", "card_0"]
        assert json.loads(payload["state"][0]) == {"landlord_leasing_definition": "Landlord office-leasing perspective: less competing vacant/available space, stronger occupier demand, and higher comparable achieved rents improve conditions; the opposites worsen them. Stable components do not reverse a change. Preserve each fact's scope, period, status and qualifications.", "card_0": {"fact": card.text, "qualifications": card.qualifications}}
        return httpx.Response(200, json={"answers": {"card_0": {"type": "choice", "choice": option, "probabilities": probabilities, "confidence": 0.73}}})
    reply = OpenRouterProvider(Settings(openrouter_api_key="synthetic"), httpx.MockTransport(respond)).classify_signals([card], 1)
    choice = reply.choices[card.fingerprint]
    assert choice.choice == label and choice.confidence == 0.73
    assert choice.probabilities == {"bullish": probabilities[options[0]], "bearish": probabilities[options[1]], "neutral": probabilities[options[2]]}


@pytest.mark.parametrize("defect", ["missing", "extra", "old_keys", "unknown_choice", "argmax", "confidence"])
def test_semantic_signal_parser_rejects_invalid_distribution(defect):
    from london_market.config import Settings
    answer = {"type": "choice", "choice": "improves_landlord_leasing_conditions", "probabilities": {
        "improves_landlord_leasing_conditions": 0.8, "worsens_landlord_leasing_conditions": 0.1, "no_directional_leasing_effect": 0.1}, "confidence": 0.73}
    if defect == "missing": del answer["probabilities"]["no_directional_leasing_effect"]
    elif defect == "extra": answer["probabilities"]["other"] = 0
    elif defect == "old_keys": answer["probabilities"] = {"bullish": 0.8, "bearish": 0.1, "neutral": 0.1}
    elif defect == "unknown_choice": answer["choice"] = "bullish"
    elif defect == "argmax": answer["choice"] = "worsens_landlord_leasing_conditions"
    else: answer["confidence"] = 2
    provider = OpenRouterProvider(Settings(openrouter_api_key="synthetic"), httpx.MockTransport(lambda request: httpx.Response(200, json={"answers": {"card_0": answer}})))
    with pytest.raises(ProviderError, match="Invalid typed signal"):
        provider.classify_signals([SignalCard(fingerprint="mock", text="Fact", qualifications=[])], 1)
