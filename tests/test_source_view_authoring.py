"""Fresh source-view input cannot manufacture commentary from measurements."""
import json
from pathlib import Path

import pytest

from london_market.config import Settings
from london_market.domain import Claim, Document, Evidence, Observation, Task, Usage
from london_market.providers import FixtureWebProvider, ModelReply, ProviderError, ToolCall
from london_market.runner import Runner
from london_market.store import Store


PACKET = Path(__file__).parents[1] / 'evidence/acceptance/requirements-coverage/build-20261011-v6-ui-v28-chat-packet.json'


def ingest_research_sources(store):
    packet = json.loads(PACKET.read_text())
    documents = [Document.model_validate(row['body']) for row in packet['documents']]
    evidence = [Evidence.model_validate(row['body']) for row in packet['evidence']]
    for document in documents:
        store.ingest(document, [item for item in evidence if item.document_id == document.id], [])
    observations = [Observation.model_validate(row['body']).model_copy(update={'evidence_ids': sorted(link['evidence_id'] for link in packet['observation_evidence'] if link['observation_id'] == row['id'])}) for row in packet['observations']]
    store.ingest(documents[0], [], observations)
    return packet['run']['result']['claims'][6]




COMMENTARY = {
    'knight-frank-london-commentary-october-2026': (
        'Knight Frank', 'London offices',
        'In London offices, development activity is not keeping pace with occupier demand for the type of space now required, a dynamic driven partly by viability pressures and partly by the same macroeconomic caution weighing on investors.',
        'Attributed Knight Frank commentary by William Matthews, in an episode with Shabab Qadar, Head of London Offices Research. Interpretation, not a quantified observed London-office indicator. No measurement reference period or independently established causal explanation.',
        ['demand'],
    ),
    'gla-london-economy': (
        'Greater London Authority', 'London',
        'With the official London unemployment rate having hit 7.4% in March 2026, this suggests that weaker demand in the labour market is the key contributor recently to involuntary worklessness in London.',
        'GLA commentary attributes the unemployment rate to official statistics; exact three-month survey window is unspecified, so no normalized observation is created. The weaker-demand explanation is attributed interpretation.',
        ['employment', 'macro'],
    ),
}


def source_view_intent(evidence_id):
    return {'kind': 'interpretation', 'interpretation_role': 'source_view', 'source_view': {'evidence_id': evidence_id}}


def commentary_source(store, family):
    from london_market.sources import MONITORED_SOURCES
    span = COMMENTARY[family][2]
    url = MONITORED_SOURCES[family].url
    return next(item for item in store.list_evidence('live') if str(item.source_url).rstrip('/') == url.rstrip('/') and item.excerpt == span and item.validation_status == 'validated')




def commentary_variant(store, original, **changes):
    """A distinct retained alias keeps the original artifact and row untouched."""
    from hashlib import sha256
    document = store.get_document(original.document_id, original.mode)
    mode = changes.get('mode', original.mode)
    document = Document.model_validate({**document.model_dump(exclude={'id'}), 'mode': mode, 'synthetic': mode == 'fixture', 'url': changes.get('source_url', document.url), 'published_at': changes.get('published_at', document.published_at), 'content_hash': sha256((original.id + str(changes)).encode()).hexdigest()})
    evidence = Evidence.model_validate({**original.model_dump(exclude={'id'}), **changes, 'document_id': document.id})
    store.ingest(document, [evidence], [])
    return store.get_evidence(evidence.id, mode)




@pytest.mark.parametrize('field,value', [('text', 'The market is robust.'), ('headline', 'Robust demand'), ('evidence_ids', ['another']), ('chart_ids', ['chart_unprepared']), ('topic_tags', ['supply']), ('span', 'Selected sentence only')])
def test_source_view_intent_excludes_free_overrides(field, value):
    from london_market.runner import FinishArgs, ReviseFinishArgs
    claim = {**source_view_intent('exact-source'), field: value}
    with pytest.raises(ValueError):
        FinishArgs(claims=[claim])
    with pytest.raises(ValueError):
        ReviseFinishArgs(draft_id='latest', replacements=[{'index': 0, 'claim': claim}])
