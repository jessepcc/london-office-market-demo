"""Rejected drafts expose all bounded diagnostics without accepting a prefix."""
import json

import pytest

from london_market.config import Settings
from london_market.domain import Task, Usage
from london_market.providers import FixtureWebProvider, ModelReply, ToolCall
from london_market.runner import CompletionValidationError, FinishArgs, ReportFinishArgs, Runner, parse_completion_args
from london_market.store import Store


# Literal vocabulary from the rejected ordinary 34-card draft: fourteen cards,
# nineteen invalid fields. This fixture deliberately preserves every mistake.
DRAFT_TAGS = [
    ['supply', 'demand'], ['supply', 'demand'], ['news', 'refurbishment', 'demand'],
    ['refurbishment', 'demand'], ['refurbishment', 'demand'], ['news', 'refurbishment'],
    ['refurbishment', 'news'], ['leasing'], ['vacancy', 'city'], ['rents', 'city', 'west_end'],
    ['rents', 'west_end', 'city'], ['rents'], ['supply', 'refurbishment'], ['supply'], ['supply'],
    ['prelets', 'leasing', 'city'], ['demand', 'leasing'], ['ESG', 'rents'], ['ESG', 'rents'],
    ['news', 'ESG', 'canary_wharf'], ['ESG', 'canary_wharf'], ['macro', 'regulations'],
    ['macro'], ['employment'], ['employment', 'macro'], ['macro', 'economy'], ['hybrid-work'],
    ['vacancy', 'midtown_fringe', 'wider_london'], ['wider_london', 'midtown_fringe'],
    ['news', 'supply'], ['supply', 'wider_london'], ['supply', 'wider_london'],
    ['supply', 'news'], ['vacancy', 'leasing'],
]


def draft(count):
    claims = [{'kind': 'watchpoint', 'text': 'Inspect the next source release for a supported market update.', 'topic_tags': tags} for tags in DRAFT_TAGS]
    claims.extend({'kind': 'watchpoint', 'text': 'Inspect the next source release.', 'topic_tags': ['supply']} for _ in range(count - len(claims)))
    return {'claims': claims, 'coverage': {'scope': 'full_market', 'gaps': {}}}


@pytest.mark.parametrize('schema,count,maximum', [(FinishArgs, 34, 24), (ReportFinishArgs, 41, 40)])
def test_count_and_all_fourteen_invalid_cards_are_returned_together(schema, count, maximum):
    with pytest.raises(CompletionValidationError) as caught:
        parse_completion_args(schema, draft(count))
    error = caught.value
    assert error.actual_claim_count == count and error.max_claim_count == maximum
    assert any(field['field'] == ['claims'] and field['type'] == 'too_long' for field in error.invalid_fields)
    tags = [field for field in error.invalid_fields if 'topic_tags' in field['field']]
    assert len(tags) == 19
    assert {field['field'][1] for field in tags} == {7, 8, 9, 10, 15, 16, 19, 20, 25, 27, 28, 30, 31, 33}


def test_excessive_diagnostic_input_is_rejected_with_explicit_incomplete_marker():
    with pytest.raises(CompletionValidationError) as caught:
        parse_completion_args(ReportFinishArgs, draft(81))
    assert caught.value.actual_claim_count == 81
    assert caught.value.diagnostic_claim_count == 80
    assert any(field['type'] == 'diagnostics_incomplete' for field in caught.value.invalid_fields)


def test_loaded_reporting_skill_exposes_forty_and_aggregates_pure_audit_before_completion(tmp_path):
    class Model:
        round = 0

        def complete(self, messages, tools, max_output_tokens, timeout):
            schema = next(item['function']['parameters'] for item in tools if item['function']['name'] == 'finish')
            if self.round == 0:
                assert schema['properties']['claims']['maxItems'] == 24
                name, args = 'load_skill', {'skill_id': 'prepare_weekly_briefing'}
            elif self.round == 1:
                assert schema['properties']['claims']['maxItems'] == 40
                assert max_output_tokens == 6000
                name, args = 'finish', draft(41)
                args['claims'][40]['text'] = 'Prepared exhibit chart_123456789abc establishes the market conclusion.'
            else:
                feedback = json.loads(next(message['content'] for message in reversed(messages) if message['role'] == 'tool'))
                assert feedback['actual_claim_count'] == 41 and feedback['max_claim_count'] == 40
                assert len([field for field in feedback['invalid_fields'] if 'topic_tags' in field['field']]) == 19
                assert any(issue.startswith('claim 40: internal') for issue in feedback['audit_issues'])
                name, args = 'finish', {'response_type': 'insufficient_data', 'claims': [], 'coverage': {'scope': 'full_market', 'gaps': {}}}
            self.round += 1
            return ModelReply(model='synthetic-boundary', provider='synthetic', usage=Usage(), calls=[ToolCall(id=str(self.round), name=name, arguments=json.dumps(args))])

        def classify_signals(self, *args, **kwargs):
            raise AssertionError('Rejected drafts must not reach classification')

    with Store(tmp_path / 'db') as store:
        model = Model()
        run = Runner(Settings(), store, model=model, web=FixtureWebProvider()).run(Task(objective='A full London market report', mode='live', entry_skill='research_market'))
        assert model.round == 3 and run.status == 'partial'
        assert len(run.result.claims) == 0
        assert run.result.coverage.scope == 'full_market'
        assert [event.outcome for event in store.list_traces(run.id) if event.name == 'finish'] == ['failed', 'success']
        assert len(store.get_run(run.id).result.claims) == 0
        assert store.list_reports('live') == []
