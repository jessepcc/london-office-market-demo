"""Public retrieval regressions, without network calls or paid credentials."""

import hashlib
import io
from datetime import datetime, timezone

import httpx
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from london_market import providers
from london_market.config import Settings
from london_market.providers import ProviderError, WebProvider


def pdf_bytes(texts):
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=600, height=800)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(('BT /F1 12 Tf 50 700 Td (' + text + ') Tj ET').encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def retrieve(raw, content_type, chars=18000):
    with httpx.MockTransport(lambda request: httpx.Response(200, content=raw, headers={'content-type': content_type})) as transport:
        return WebProvider(Settings(max_document_chars=chars), transport, resolver=lambda host: ['93.184.216.34']).retrieve('https://example.com/report', 2)


def test_tavily_rfc_dates_and_unknown_discoveries_survive(monkeypatch):
    monkeypatch.setattr(providers, 'utc_now', lambda: datetime(2026, 10, 10, tzinfo=timezone.utc))
    items = [
        {'url': 'https://example.com/recent', 'published_date': 'Fri, 29 May 2026 00:00:00 GMT'},
        {'url': 'https://example.com/unknown', 'published_date': None},
        {'url': 'https://example.com/malformed', 'published_date': 'not a date'},
        {'url': 'https://example.com/old', 'published_date': 'Tue, 11 Mar 2025 17:00:00 GMT'},
    ]
    with httpx.MockTransport(lambda request: httpx.Response(200, json={'results': items})) as transport:
        result = WebProvider(Settings(tavily_api_key='private-key'), transport).search('London offices', 2)
    assert [str(hit.url) for hit in result.results] == ['https://example.com/recent', 'https://example.com/unknown', 'https://example.com/malformed']
    assert result.results[0].published_date_estimate.isoformat() == '2026-05-29'
    assert result.results[1].published_date_estimate is None
    assert result.results[2].published_date_estimate is None
    assert any('Retained 2' in caveat and 'unknown' in caveat for caveat in result.caveats)
    assert any('Excluded 1' in caveat for caveat in result.caveats)


@pytest.mark.parametrize('kind', ['search', 'retrieve'])
def test_http_status_diagnostics_do_not_leak_credentials_or_body(kind):
    with httpx.MockTransport(lambda request: httpx.Response(429, text='secret-body private-key')) as transport:
        provider = WebProvider(Settings(tavily_api_key='private-key'), transport, resolver=lambda host: ['93.184.216.34'])
        with pytest.raises(ProviderError) as caught:
            provider.search('London offices', 2) if kind == 'search' else provider.retrieve('https://example.com/report?token=private-key', 2)
    assert str(caught.value) == ('Tavily HTTP 429' if kind == 'search' else 'Document HTTP 429')


def test_html_main_content_precedes_navigation_and_retains_marked_publication():
    raw = b'<html><title>Quarterly market</title><article>Navigation promo</article><header>Navigation</header><main><h1>London offices</h1><time itemprop="datePublished" datetime="2026-05-14">14 May 2026</time><article><h2>Take-up</h2><p>Take-up reached 2 million sq ft.</p></article></main><footer>Contact and cookies</footer></html>'
    page = retrieve(raw, 'text/html')
    assert page.text.startswith('London offices')
    assert 'Take-up reached 2 million sq ft.' in page.text
    assert 'Navigation' not in page.text and 'cookies' not in page.text
    assert page.document.published_at.isoformat() == '2026-05-14'
    assert page.document.content_hash == hashlib.sha256(raw).hexdigest()


def test_publication_is_not_guessed_from_arbitrary_body_dates():
    page = retrieve(b'<main><p>Q1 report discussed on 14 May 2026.</p><time datetime="2026-05-14">Event date</time></main>', 'text/html')
    assert page.document.published_at is None


@pytest.mark.parametrize('markup', [
    '<meta property="article:published_time" content="2026-05-14T10:20:00Z">',
    '<script type="application/ld+json">{"@type":"NewsArticle","datePublished":"2026-05-14","dateModified":"2026-05-15"}</script>',
    '<time class="sv-content-header__date">14 May 2026</time>',
])
def test_authoritative_publication_markup_preserves_publication_not_update_date(markup):
    page = retrieve(('<main>' + markup + '<p>Office research.</p></main>').encode(), 'text/html')
    assert page.document.published_at.isoformat() == '2026-05-14'


def test_pdf_text_and_full_document_identity_survive_character_truncation():
    raw = pdf_bytes(['Office take-up reached 2 million sq ft. ' * 50, 'London office investment.'])
    short = retrieve(raw, 'application/pdf', chars=1000)
    full = retrieve(raw, 'application/pdf', chars=4000)
    assert short.text.startswith('[Page 1]')
    assert 'Office take-up reached 2 million sq ft.' in short.text
    assert len(short.text) == 1000 and short.truncated
    assert full.document.id == short.document.id
    assert short.document.content_hash == hashlib.sha256(raw).hexdigest()
    assert short.document.published_at is None
    assert full.total_pages == 2 and full.extracted_pages == 2
    assert not full.truncated and '[Page 2]' in full.text
    assert any('character limit' in caveat for caveat in short.caveats)


def test_pdf_page_bound_is_explicit_and_does_not_claim_full_extraction():
    page = retrieve(pdf_bytes([f'Office page {index}.' for index in range(1, 62)]), 'application/pdf', chars=100000)
    assert page.total_pages == 61 and page.extracted_pages == 60
    assert page.truncated and '[Page 60]' in page.text and '[Page 61]' not in page.text
    assert any('page limit' in caveat for caveat in page.caveats)


def test_pdf_size_and_non_text_limits_are_explicit():
    with httpx.MockTransport(lambda request: httpx.Response(200, headers={'content-type': 'application/pdf', 'content-length': str(8 * 1024 * 1024 + 1)}, content=b'%PDF')) as transport:
        with pytest.raises(ProviderError, match='size limit'):
            WebProvider(Settings(), transport, resolver=lambda host: ['93.184.216.34']).retrieve('https://example.com/report', 2)
    with pytest.raises(ProviderError, match='PDF contains no extractable text'):
        retrieve(pdf_bytes(['']), 'application/pdf')
    with pytest.raises(ProviderError, match='Invalid or unsupported PDF'):
        retrieve(b'%PDF-invalid', 'application/pdf')


def test_plain_text_is_not_parsed_as_html():
    page = retrieve(b'Office rents < GBP150 per sq ft; unknown publication.', 'text/plain')
    assert page.text == 'Office rents < GBP150 per sq ft; unknown publication.'
