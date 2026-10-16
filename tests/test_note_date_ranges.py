import json
from pathlib import Path

import pytest

from london_market.claim_audit import note_figures
from london_market.runner import completion_metadata_audit




@pytest.mark.parametrize('note',[
    'No release during 5–11 October 2026.',
    'No release during 5-11 Oct 2026.',
    'No release during 5 — 11 October.',
    'No release during 28–29 February 2024.',
])
def test_calendar_ranges_do_not_add_statistics(note):
    assert all(1900<=value<=2100 for value in note_figures(note))


@pytest.mark.parametrize('note,expected',[
    ('Vacancy is 5–11% in October 2026.',{5,11,2026}),
    ('Reported supply is 5–11 million sq ft.',{5,11}),
    ('No release in the 5–11 October 2026 window; supply declined 0.4 million sq ft and 15 schemes remain.',{.4,15}),
    ('No release during 32–33 October 2026.',{32,33,2026}),
    ('No release during 28–29 February 2025.',{28,29,2025}),
    ('No release during 11–5 October 2026.',{11,5,2026}),
    ('No release during 5–11 Smarch 2026.',{5,11,2026}),
])
def test_quantity_ranges_and_invalid_calendar_ranges_remain_auditable(note,expected):
    assert note_figures(note)==expected
