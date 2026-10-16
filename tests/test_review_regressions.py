"""Review regressions for growing evidence, chronology and actual HTTP deadlines."""

import asyncio
import hashlib
import time
import threading
from calendar import monthrange
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from pydantic import ValidationError

from london_market.config import Settings
from london_market.domain import Document, Evidence, Observation, RunRecord, Task
from london_market.providers import FixtureWebProvider, OpenRouterProvider, ProviderError, WebProvider, post_json
from london_market.store import Store
from london_market.tools import QueryArgs, ToolContext, VisualizationArgs, choose_visualization, query_evidence


def context(store):
    run = RunRecord(task=Task(objective="London offices", entry_skill="research_market"), main_model="synthetic", router_model="synthetic")
    store.save_run(run)
    return ToolContext(Settings(), store, FixtureWebProvider(), "fixture", run.id, 2)


def ingest(store, start, end, value, *, published=date(2026, 10, 1), geography="City", metric="vacancy_rate"):
    text = f"Synthetic {geography} {metric} {value}% from {start} to {end}."
    document = Document(mode="fixture", synthetic=True, url=f"https://fixtures.example.invalid/{start}/{value}/{geography}/{metric}", title="Synthetic review source", publisher="Synthetic Market Publisher", content_hash=hashlib.sha256(text.encode()).hexdigest(), published_at=published, retrieved_at=datetime(2026, 10, 9, tzinfo=timezone.utc))
    evidence = Evidence(document_id=document.id, mode="fixture", excerpt=text, source_url=document.url, native_geography=geography, mapped_geography=geography, category="vacancy" if metric == "vacancy_rate" else "availability", published_at=published, retrieved_at=document.retrieved_at, validation_status="validated", validation_reason="Controlled synthetic regression data")
    observation = Observation(mode="fixture", metric=metric, value=value, unit="%", publisher=document.publisher, definition="Synthetic stable definition", category=evidence.category, native_geography=geography, mapped_geography=geography, period_start=start, period_end=end, evidence_ids=[evidence.id])
    store.ingest(document, [evidence], [observation])
    return observation, evidence


def test_recent_periods_and_pagination_do_not_hide_thirteenth_measurement(tmp_path):
    with Store(tmp_path / "db") as store:
        ctx = context(store)
        inserted = []
        # Publication order deliberately contradicts reference-period order.
        for index in range(13):
            year, month = 2025 + index // 12, index % 12 + 1
            inserted.append(ingest(store, date(year, month, 1), date(year, month, monthrange(year, month)[1]), index + 1, published=date(2026, 10, 9) if index == 0 else date(2026, 9, 1)))
        ingest(store, date(2026, 2, 1), date(2026, 2, 28), 50, geography="West End")
        ingest(store, date(2026, 2, 1), date(2026, 2, 28), 60, metric="availability_rate")
        first = query_evidence(ctx, QueryArgs(geography="City", metric="vacancy_rate", limit=12))
        assert first.observations[0].id == inserted[-1][0].id
        assert first.observation_total == first.evidence_total == 13
        assert first.next_offset == 12 and first.truncated
        assert {item.id for item in first.evidence} >= {item.evidence_ids[0] for item in first.observations}
        assert all(item.category == "vacancy" and item.mapped_geography == "City" for item in first.evidence)
        last = query_evidence(ctx, QueryArgs(geography="City", metric="vacancy_rate", limit=12, offset=12))
        assert [item.id for item in last.observations] == [inserted[0][0].id]
        assert last.next_offset is None
        assert last.truncated is False
        assert {item.id for item in last.evidence} >= set(last.supporting_evidence_ids)
        assert len({item.id for item in first.observations + last.observations}) == 13
        beyond = query_evidence(ctx, QueryArgs(geography="City", metric="vacancy_rate", offset=13))
        assert beyond.observations == beyond.evidence == [] and beyond.next_offset is None
        assert beyond.truncated is False
    with pytest.raises(ValidationError):
        QueryArgs(offset=-1)
    with pytest.raises(ValidationError):
        QueryArgs(limit=13)


def test_small_page_flags_conflicts_outside_the_page_and_keeps_support(tmp_path):
    with Store(tmp_path / "db") as store:
        ctx = context(store)
        a, _ = ingest(store, date(2026, 1, 1), date(2026, 1, 31), 8)
        b, _ = ingest(store, date(2026, 1, 1), date(2026, 1, 31), 9)
        result = query_evidence(ctx, QueryArgs(metric="vacancy_rate", limit=1))
        assert len(result.observations) == 1
        assert result.conflicts[0].semantic_key == a.semantic_key
        assert set(result.conflicts[0].observation_ids) == {a.id, b.id}
        assert result.supporting_evidence_ids[0] in {item.id for item in result.evidence}


def test_reference_and_publication_orders_remain_distinct_with_exact_metric_support(tmp_path):
    with Store(tmp_path / "db") as store:
        ctx = context(store)
        recent, recent_source = ingest(store, date(2026, 4, 1), date(2026, 4, 30), 7, published=date(2026, 5, 1))
        older, older_source = ingest(store, date(2026, 1, 1), date(2026, 1, 31), 8, published=date(2026, 10, 8))
        ingest(store, date(2026, 6, 1), date(2026, 6, 30), 10, metric="available_space")
        result = query_evidence(ctx, QueryArgs(metric="vacancy_rate", limit=1))
        assert [item.id for item in result.observations] == [recent.id]
        assert [item.id for item in result.evidence] == [older_source.id, recent_source.id]
        assert result.supporting_evidence_ids == [recent_source.id]
        assert result.observation_total == result.evidence_total == 2
        assert query_evidence(ctx, QueryArgs(metric="availability_rate")).evidence == []
        assert query_evidence(ctx, QueryArgs(metric="nonexistent", offset=4)).truncated is False


@pytest.mark.parametrize("periods,reason", [
    ([(date(2026, 1, 1), date(2026, 1, 31)), (date(2026, 4, 1), date(2026, 4, 30))], "Missing"),
    ([(date(2026, 1, 1), date(2026, 1, 10)), (date(2026, 1, 11), date(2026, 1, 20)), (date(2026, 1, 18), date(2026, 1, 27))], "overlapping"),
])
def test_trend_checks_adjacent_periods_not_just_first(tmp_path, periods, reason):
    with Store(tmp_path / "db") as store:
        ctx = context(store)
        items = [ingest(store, start, end, index + 1)[0] for index, (start, end) in enumerate(periods)]
        plan = choose_visualization(ctx, VisualizationArgs(observation_ids=[item.id for item in reversed(items)], title="Synthetic trend", purpose="Review chronology")).plan
        assert plan.chart_type == "table"
        assert any(reason in caveat for caveat in plan.caveats)
        assert plan.observation_ids == [item.id for item in items]


class Trickle(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        for chunk in [b'{"results":', b'[]', b'}']:
            await asyncio.sleep(.025)
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("kind", ["openrouter_headers", "openrouter_body", "tavily_headers", "tavily_body", "document_headers", "document_body"])
def test_http_deadline_cancels_slow_headers_or_trickle_body(kind):
    stream = Trickle()
    cancelled = []

    async def handler(request):
        if kind.endswith("headers"):
            try:
                await asyncio.sleep(.2)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
        return httpx.Response(200, headers={"content-type": "text/plain"}, stream=stream)

    transport = httpx.MockTransport(handler)
    settings = Settings(openrouter_api_key="synthetic-key", tavily_api_key="synthetic-key")
    provider = OpenRouterProvider(settings, transport=transport) if kind.startswith("openrouter") else WebProvider(settings, transport=transport, resolver=lambda host: ["93.184.216.34"])
    start = time.monotonic()
    with pytest.raises(ProviderError, match="timeout|deadline"):
        if kind.startswith("openrouter"):
            provider.complete([], [], 50, .04)
        elif kind.startswith("tavily"):
            provider.search("London offices", .04)
        else:
            provider.retrieve("https://example.com/", .04)
    assert time.monotonic() - start < .15
    assert cancelled if kind.endswith("headers") else stream.closed
    provider.close()


def test_json_body_size_is_bounded_and_real_pooled_transport_is_not_reused():
    with pytest.raises(ValueError, match="MockTransport"):
        OpenRouterProvider(Settings(), transport=httpx.AsyncHTTPTransport())
    provider = OpenRouterProvider(Settings(openrouter_api_key="synthetic"), transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b" " * 1_000_001)))
    with pytest.raises(ProviderError, match="size limit"):
        provider.complete([], [], 50, 1)


@pytest.mark.parametrize("kind", ["openrouter", "tavily", "document"])
def test_identity_encoding_rejects_compression_before_body_read(kind):
    read = []
    class Unread(httpx.AsyncByteStream):
        async def __aiter__(self):
            read.append(True)
            yield b"compressed-body"
    def handler(request):
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(200, headers={"content-encoding": "gzip", "content-type": "text/plain"}, stream=Unread())
    transport = httpx.MockTransport(handler)
    settings = Settings(openrouter_api_key="synthetic", tavily_api_key="synthetic")
    with pytest.raises(ProviderError, match="identity"):
        if kind == "openrouter":
            OpenRouterProvider(settings, transport).complete([], [], 50, 1)
        elif kind == "tavily":
            WebProvider(settings, transport).search("London offices", 1)
        else:
            WebProvider(settings, transport, resolver=lambda host: ["93.184.216.34"]).retrieve("https://example.com", 1)
    assert read == []


def test_redirects_share_one_deadline_and_transport_supports_repeated_calls():
    seen = []
    async def redirects(request):
        seen.append(str(request.url))
        await asyncio.sleep(.025)
        return httpx.Response(302, headers={"location": "/next"})
    web = WebProvider(Settings(), httpx.MockTransport(redirects), resolver=lambda host: ["93.184.216.34"])
    with pytest.raises(ProviderError, match="deadline"):
        web.retrieve("https://example.com/first", .04)
    assert len(seen) == 2
    provider = OpenRouterProvider(Settings(openrouter_api_key="synthetic"), httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "Typed result", "tool_calls": []}}]})))
    assert provider.complete([], [], 50, 1).content == "Typed result"
    assert provider.complete([], [], 50, 1).content == "Typed result"
    provider.close()


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_default_http_transport_cancels_actual_slow_local_server(monkeypatch, phase):
    # Only the test server runs in a thread; the client request is cancellable
    # async I/O in the calling thread, with no detached client worker.
    received = threading.Event()
    disconnected = threading.Event()
    class SlowServer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            received.set()
            try:
                if phase == "headers":
                    time.sleep(.6)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                for chunk in b'{"results":[]}':
                    self.wfile.write(bytes([chunk]))
                    self.wfile.flush()
                    time.sleep(.05)
            except (BrokenPipeError, ConnectionResetError):
                disconnected.set()
    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowServer)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    try:
        start = time.monotonic()
        with pytest.raises(ProviderError, match="deadline"):
            post_json(f"http://127.0.0.1:{server.server_port}/", {}, .2, None)
        assert time.monotonic() - start < .5
        assert received.is_set()
        assert disconnected.wait(1)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)
