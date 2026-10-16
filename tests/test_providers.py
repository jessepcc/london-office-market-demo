"""Real API wire contracts and pre-call limits exercised with HTTP mock transports."""

import hashlib
import json
import socket

import httpx
import pytest

from london_market.config import Settings
from london_market.domain import Document, Evidence, Limits, RunRecord, Task
from london_market.providers import OpenRouterProvider, ProviderError, WebProvider, public_url, router_state
from london_market.runner import Runner, rates
from london_market.store import Store
from london_market.tools import MONITORING_URL, NormalizeArgs, ToolContext, normalize_observation


def completion(name, arguments, cost=.0001):
    return {"model": "deepseek/deepseek-v4.1-flash", "provider": "mock-provider", "choices": [{"message": {"role": "assistant", "content": None, "reasoning": "hidden-reasoning-must-not-persist", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": cost}}


def route_response():
    return {"model": "cloudflare/clef-flash", "provider": "Cloudflare", "answers": {"route": {"type": "choice", "choice": "in_scope", "probabilities": {"in_scope": .98, "ambiguous": .01, "unrelated": .01, "greeting": 0}, "confidence": .96}}, "usage": {"input_tokens": 289, "output_tokens": 0, "cost": .00002601}}


def test_decisions_and_chat_completions_wire_contracts():
    seen = []
    def handler(request):
        payload = json.loads(request.content)
        seen.append((request.url.path, payload))
        assert request.headers["Authorization"] == "Bearer mock-key"
        if request.url.path.endswith("decisions"):
            return httpx.Response(200, json=route_response())
        return httpx.Response(200, json=completion("query_evidence", {}))
    with httpx.MockTransport(handler) as client:
        provider = OpenRouterProvider(Settings(openrouter_api_key="mock-key"), client)
        question = Task(objective="What about Canary Wharf?", entry_skill="research_market", history=[])
        route = provider.route(question, 2)
        assert route.route.choice == "in_scope" and route.usage.spend_usd == .00002601
        reply = provider.complete([{"role": "user", "content": "London offices"}], [], 900, 2)
        assert reply.calls[0].name == "query_evidence"
        assert reply.provider == "mock-provider"
    assert seen[0][0] == "/api/alpha/decisions"
    assert seen[0][1]["model"] == "cloudflare/clef-flash"
    assert isinstance(seen[0][1]["state"][0], str)
    assert seen[0][1]["state"][0].startswith("Newest question: What about Canary Wharf?")
    assert seen[0][1]["questions"]["route"]["criteria"] == {"in_scope": None, "ambiguous": None, "unrelated": None, "greeting": None}
    assert seen[1][0] == "/api/v1/chat/completions"
    assert seen[1][1]["model"] == "deepseek/deepseek-v4.1-flash" and seen[1][1]["max_tokens"] == 900
    assert seen[1][1]["tool_choice"] == "required"
    assert seen[1][1]["reasoning"] == {"enabled": False, "exclude": True}


def test_reasoning_usage_diagnostics_never_store_hidden_content_or_double_count(tmp_path):
    body = completion("finish", {})
    body["choices"][0]["finish_reason"] = "length"
    body["choices"][0]["message"].update(content="uncited-market-prose", tool_calls=[], reasoning="hidden-reasoning-must-not-persist")
    body["usage"].update(completion_tokens=4000, completion_tokens_details={"reasoning_tokens": 3900})
    settings = Settings(mode="live", openrouter_api_key="mock-key")
    with httpx.MockTransport(lambda request: httpx.Response(200, json=body)) as transport:
        provider = OpenRouterProvider(settings, transport)
        reply = provider.complete([], [], 4000, 2)
        assert reply.reasoning_tokens == 3900
        assert reply.unstructured_content_characters == len("uncited-market-prose")
        with Store(tmp_path / "db") as store:
            run = Runner(settings, store, model=provider).run(Task(objective="Research London offices", entry_skill="research_market", trigger="manual", mode="live"))
            assert run.status == "failed"
            assert run.usage.output_tokens == 4000
            response = next(event for event in store.list_traces(run.id) if event.kind == "model" and event.outcome == "response")
            assert response.data["reasoning_tokens"] == 3900
            assert response.data["unstructured_content_characters"] == len("uncited-market-prose")
            persisted = json.dumps([event.model_dump(mode="json") for event in store.list_traces(run.id)]) + run.model_dump_json()
            assert "hidden-reasoning-must-not-persist" not in persisted
            assert "uncited-market-prose" not in persisted


def test_router_context_conservative_bound_and_invalid_typed_response():
    from london_market.domain import ChatMessage
    question = Task(objective="What about Canary Wharf?", entry_skill="research_market", history=[ChatMessage(role="user", content="London office leasing " + "x" * 5000)])
    state = router_state(question)
    assert len(state.encode()) <= 1500
    assert state.startswith("Newest question: What about Canary Wharf?")
    with httpx.MockTransport(lambda request: httpx.Response(200, json={"answers": {"route": {"type": "noul"}}})) as client:
        with pytest.raises(ProviderError, match="Invalid typed"):
            OpenRouterProvider(Settings(openrouter_api_key="mock"), client).route(question, 2)


def test_provider_errors_never_expose_credentials_or_bodies():
    with httpx.MockTransport(lambda request: httpx.Response(401, text="private-key-from-response")) as client:
        with pytest.raises(ProviderError) as caught:
            OpenRouterProvider(Settings(openrouter_api_key="private-request-key"), client).complete([], [], 900, 1)
    assert str(caught.value) == "OpenRouter HTTP 401"
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("url", ["http://127.0.0.1", "http://169.254.169.254/latest/meta-data", "http://localhost", "http://[::1]", "http://private.internal", "https://user:password@example.com", "file:///tmp/secret", "https://example.com:8443/"])
def test_private_urls_rejected_before_http(url):
    with pytest.raises(ProviderError):
        public_url(url, resolver=lambda host: [host] if host.replace(".", "").isdigit() else ["93.184.216.34"])


def test_redirect_to_private_and_oversize_are_rejected():
    seen = []
    def redirect(request):
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})
    with httpx.MockTransport(redirect) as client:
        with pytest.raises(ProviderError, match="Private"):
            WebProvider(Settings(), client, resolver=lambda host: ["93.184.216.34"]).retrieve("https://example.com/", 2)
    assert seen == ["https://example.com/"]
    with httpx.MockTransport(lambda request: httpx.Response(200, headers={"content-type": "text/html", "content-length": "256001"}, text="x")) as client:
        with pytest.raises(ProviderError, match="size limit"):
            WebProvider(Settings(), client, resolver=lambda host: ["93.184.216.34"]).retrieve("https://example.com/", 2)


def test_proxy_fallback_only_verified_public_host(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://managed-proxy.invalid")
    def no_local_dns(*args, **kwargs):
        raise socket.gaierror("cloud local DNS unavailable")
    monkeypatch.setattr(socket, "getaddrinfo", no_local_dns)
    assert public_url(MONITORING_URL) == MONITORING_URL
    with pytest.raises(ProviderError, match="public destination"):
        public_url("https://unvalidated-source.invalid")
    monkeypatch.delenv("HTTPS_PROXY")
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("ALL_PROXY", raising=False)
    monkeypatch.delenv("all_proxy", raising=False)
    with pytest.raises(ProviderError):
        public_url(MONITORING_URL)


def test_retrieval_retains_unknown_publication_and_marks_truncation():
    html = "<html><title>Office source</title><body>" + "London office context. " * 200 + "</body></html>"
    with httpx.MockTransport(lambda request: httpx.Response(200, headers={"content-type": "text/html"}, text=html)) as client:
        page = WebProvider(Settings(max_document_chars=1000), client, resolver=lambda host: ["93.184.216.34"]).retrieve("https://example.com/offices", 2)
        assert len(page.text) == 1000 and page.truncated
        assert page.document.published_at is None
        assert page.document.retrieved_at.tzinfo is not None
        assert page.document.mode == "live"


def test_live_http_mock_research_full_tool_roundtrip_under_defaults(tmp_path):
    call_count = [0]
    paths = []
    excerpt = "City office vacancy was 8.0% in Q1 2026."
    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith("decisions"):
            return httpx.Response(200, json=route_response())
        if request.url.host == "api.tavily.com":
            body = json.loads(request.content)
            assert body["search_depth"] == "basic" and body["api_key"] == "search-key"
            return httpx.Response(200, json={"results": [{"url": "https://example.com/offices", "title": "Office data", "content": "Discovery snippet", "published_date": "2026-10-09"}]})
        if request.url.host == "example.com":
            return httpx.Response(200, headers={"content-type": "text/html"}, text="<title>Office data</title><p>" + excerpt + "</p>")
        body = json.loads(request.content)
        call_count[0] += 1
        tool_outputs = [json.loads(item["content"]) for item in body["messages"] if item["role"] == "tool"]
        actions = [("query_evidence", {}), ("search", {"query": "City office vacancy"}), ("retrieve", {"url": "https://example.com/offices"})]
        if call_count[0] <= 3:
            name, args = actions[call_count[0]-1]
        elif call_count[0] == 4:
            name, args = "cache_evidence", {"retrieved_id": tool_outputs[-1]["retrieved_id"], "excerpt": excerpt, "native_geography": "City", "mapped_geography": "City", "category": "vacancy"}
        else:
            name, args = "finish", {"claims": [{"kind": "fact", "text": excerpt, "evidence_ids": [tool_outputs[-1]["evidence"]["id"]]}], "limitations": ["Quarterly historical evidence; wider London absent"], "topic_gaps": ["Rents", "Pipeline"]}
        return httpx.Response(200, json=completion(name, args))
    settings = Settings(mode="live", openrouter_api_key="model-key", tavily_api_key="search-key")
    with Store(tmp_path / "db") as store, httpx.MockTransport(handler) as client:
        model = OpenRouterProvider(settings, client)
        web = WebProvider(settings, client, resolver=lambda host: ["93.184.216.34"])
        run = Runner(settings, store, model=model, web=web).run(Task(objective="What is City office vacancy?", entry_skill="research_market", mode="live", trigger="chat"))
        assert run.status == "partial"
        assert call_count[0] == 5
        assert run.result.claims[0].text == excerpt
        assert store.list_traces(run.id)[-1].data["steps"] == 12
        assert run.result.usage.spend_usd >= .0085 and not run.result.usage.cost_known
        assert "hidden-reasoning-must-not-persist" not in str(store.list_traces(run.id))
        assert "model-key" not in run.model_dump_json()
        assert store.list_evidence("live")[0].validation_status == "candidate"
        assert store.list_observations("live") == []
        assert store.list_reports("live") == []
    assert "/api/alpha/decisions" in paths and "/search" in paths


@pytest.mark.parametrize("limits,model_name,expected", [(Limits(max_tokens=1), "deepseek/deepseek-v4.1-flash", "Token limit"), (Limits(max_spend_usd=.000001), "deepseek/deepseek-v4.1-flash", "Spending limit"), (Limits(), "unknown/model", "pricing is unknown")])
def test_budget_reservation_blocks_paid_request_before_http(tmp_path, limits, model_name, expected):
    requests = []
    def handler(request):
        requests.append(request)
        raise AssertionError("paid request must be blocked")
    settings = Settings(mode="live", main_model=model_name, openrouter_api_key="mock")
    with Store(tmp_path / "db") as store, httpx.MockTransport(handler) as client:
        run = Runner(settings, store, model=OpenRouterProvider(settings, client)).run(Task(objective="London offices", entry_skill="research_market", mode="live", limits=limits))
        assert run.status == "stopped" and expected in run.result.reason
    assert requests == []


def test_selected_main_model_has_budget_reservation_rates():
    settings = Settings()
    assert settings.main_model == "deepseek/deepseek-v4.1-flash"
    assert rates(settings, "main") == (0.0000003, 0.0000012)


def test_unknown_cost_charged_at_reserved_bound(tmp_path):
    requested_output = []
    def handler(request):
        requested_output.append(json.loads(request.content)["max_tokens"])
        return httpx.Response(200, json=completion("finish", {"response_type": "insufficient_data", "message": "No sufficient source evidence"}, cost=None))
    settings = Settings(mode="live", openrouter_api_key="mock")
    with Store(tmp_path / "db") as store, httpx.MockTransport(handler) as client:
        run = Runner(settings, store, model=OpenRouterProvider(settings, client)).run(Task(objective="London offices", entry_skill="research_market", mode="live"))
        assert run.status == "partial"
        assert not run.result.usage.cost_known and run.result.usage.spend_usd > 0
        assert run.result.usage.input_tokens > 1000 and run.result.usage.output_tokens == requested_output[0]


def test_search_spend_reserved_before_tavily_request(tmp_path):
    search_requests = []
    def handler(request):
        if request.url.host == "api.tavily.com":
            search_requests.append(request)
            raise AssertionError("search must be blocked")
        return httpx.Response(200, json=completion("search", {"query": "London offices"}, cost=.045))
    settings = Settings(mode="live", openrouter_api_key="mock", tavily_api_key="mock")
    with Store(tmp_path / "db") as store, httpx.MockTransport(handler) as client:
        run = Runner(settings, store, model=OpenRouterProvider(settings, client), web=WebProvider(settings, client)).run(Task(objective="London offices", entry_skill="research_market", mode="live", limits=Limits(max_spend_usd=.05)))
        assert run.status == "stopped" and "Spending limit" in run.result.reason
    assert search_requests == []


def test_missing_credentials_still_create_inspectable_failed_run(tmp_path, monkeypatch):
    with Store(tmp_path / "db") as store:
        run = Runner(Settings(mode="live"), store).run(Task(objective="London offices", entry_skill="research_market", mode="live"))
        assert run.status == "failed"
        assert "OPENROUTER_API_KEY" in run.result.reason
        assert store.get_run(run.id).finished_at is not None


def monitored_context(store, url=MONITORING_URL):
    settings = Settings(mode="live", cw_permission=True)
    task = Task(objective="Collect monitored source", entry_skill="collect_market_evidence", mode="live")
    run = RunRecord(task=task, main_model="mock", router_model="mock")
    store.save_run(run)
    excerpt = "KEY TAKEAWAYS FOR Q2 2026 Central London office take - up totalled 2.46 million sq ft in Q2 2026. Availability was unchanged at 26.41 million sq ft in Q2. 11.62 million sq ft of space was under construction at the end of Q2."
    document = Document(mode="live", url=url, title="London Marketbeat", publisher="www.cushmanwakefield.com", content_hash=hashlib.sha256(excerpt.encode()).hexdigest())
    evidence = Evidence(mode="live", document_id=document.id, excerpt=excerpt, published_at=None, retrieved_at=document.retrieved_at, source_url=document.url, native_geography="Central London", mapped_geography="Central London", category="take-up")
    store.ingest(document, [evidence], [])
    return ToolContext(settings, store, WebProvider(settings), "live", run.id, 2), evidence


def test_live_normalization_binds_metric_value_and_approved_path(tmp_path):
    with Store(tmp_path / "db") as store:
        ctx, evidence = monitored_context(store)
        args = NormalizeArgs(evidence_id=evidence.id, metric="take_up", value=26.41, unit="million sq ft", definition="Office space leased during the reference quarter", category="take-up", native_geography="Central London", mapped_geography="Central London", period_start="2026-04-01", period_end="2026-06-30")
        with pytest.raises(ValueError, match="metric-specific"):
            normalize_observation(ctx, args)
        result = normalize_observation(ctx, args.model_copy(update={"value": 2.46}))
        assert result.observation.value == 2.46 and result.observation.mapped_geography == "Central London"
        ctx.web.close()
        sibling_ctx, sibling_evidence = monitored_context(store, MONITORING_URL + "-unvalidated")
        with pytest.raises(ValueError, match="monitoring source validation"):
            normalize_observation(sibling_ctx, args.model_copy(update={"value": 2.46, "evidence_id": sibling_evidence.id}))
        sibling_ctx.web.close()


def test_empty_choices_response_is_a_provider_error():
    with httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": []})) as client:
        with pytest.raises(ProviderError, match="Invalid Chat Completions"):
            OpenRouterProvider(Settings(openrouter_api_key="mock"), client).complete([], [], 900, 1)


def test_cushman_permission_gate_blocks_retrieval_before_http(tmp_path):
    requests = []
    def handler(request):
        requests.append(request)
        raise AssertionError("source permission gate should prevent HTTP")
    with httpx.MockTransport(handler) as client:
        with pytest.raises(ProviderError, match="reuse permission"):
            WebProvider(Settings(), client, resolver=lambda host: ["93.184.216.34"]).retrieve(MONITORING_URL, 2)
    assert requests == []
    with Store(tmp_path / "db") as store:
        ctx, evidence = monitored_context(store)
        ctx.settings = Settings(mode="live")
        args = NormalizeArgs(evidence_id=evidence.id, metric="take_up", value=2.46, unit="million sq ft", definition="Office space leased during the reference quarter", category="take-up", native_geography="Central London", mapped_geography="Central London", period_start="2026-04-01", period_end="2026-06-30")
        with pytest.raises(ValueError, match="reuse permission"):
            normalize_observation(ctx, args)
        assert store.get_evidence(evidence.id).validation_status == "candidate"
        ctx.web.close()


def test_tavily_history_window_is_requested_and_enforced(monkeypatch):
    from datetime import datetime, timezone
    from london_market import providers
    monkeypatch.setattr(providers, "utc_now", lambda: datetime(2026, 10, 9, tzinfo=timezone.utc))
    def handler(request):
        payload = json.loads(request.content)
        assert payload["start_date"] == "2026-04-13"
        assert payload["end_date"] == "2026-10-10"
        assert payload["filter_by_published_date"] is True
        assert payload["include_published_date"] is True
        return httpx.Response(200, json={"results": [
            {"url": "https://example.com/recent", "published_date": "2026-10-09", "content": "Recent update"},
            {"url": "https://example.com/old", "published_date": "2025-01-01", "content": "Older publication"},
            {"url": "https://example.com/undated", "published_date": None, "content": "Unknown freshness"},
        ]})
    with httpx.MockTransport(handler) as client:
        result = WebProvider(Settings(tavily_api_key="mock"), client).search("London offices", 2)
    assert [str(hit.url) for hit in result.results] == ["https://example.com/recent", "https://example.com/undated"]
    assert result.results[0].published_date_estimate.isoformat() == "2026-10-09"
    assert "Excluded 1" in result.caveats[1]
    assert "Retained 1" in result.caveats[2]


@pytest.mark.parametrize("ignored, expected", [("Wafer", ["Wafer"]), ("Wafer, custom-provider", ["Wafer", "custom-provider"]), ("", [])])
def test_finalization_requests_the_named_tool_from_capable_providers(monkeypatch, ignored, expected):
    from london_market.config import load_settings
    monkeypatch.setenv("OPENROUTER_MAIN_PROVIDER_IGNORE", ignored)
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-key")
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion("finish", {"response_type": "insufficient_data"}))
    with httpx.MockTransport(handler) as transport:
        provider = OpenRouterProvider(load_settings(env_file=None), transport)
        provider.complete([], [{"type": "function", "function": {"name": "finish", "parameters": {"type": "object"}}}], 900, 2)
    assert seen[0]["tool_choice"] == {"type": "function", "function": {"name": "finish"}}
    assert seen[0]["provider"] == {"require_parameters": True, "ignore": expected}


def test_invalid_provider_json_has_a_clear_bounded_error():
    from london_market.providers import post_json
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b'<html>temporary upstream error</html>'))
    with pytest.raises(ProviderError, match='invalid structured response; retained evidence remains available'):
        post_json('https://openrouter.ai/api/v1/chat/completions', {}, 1, transport)
