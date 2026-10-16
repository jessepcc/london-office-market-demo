# London office market demo

A Python research agent for exploring London office market evidence. Ask a question, collect public sources on demand, prepare a dated landlord briefing, and inspect the sources behind its claims and charts.

[Project presentation](https://jessepcc.github.io/london-office-market-demo/) · [Try the interactive demo](https://london-office-market-demo.streamlit.app/)

## Try it

The app opens without a passcode. **Synthetic demo** works without credentials or provider calls. It uses labeled example data and scripted model behavior.

For **Live with your keys**, enter your own OpenRouter and Tavily API keys in the sidebar. OpenRouter supplies model access; Tavily supplies fresh web search. You can change the research and routing model identifiers. Both services may charge your account. Free hosting does not make provider usage free.

Each browser session has its own temporary SQLite database, conversation, reports and keys. The hosted Python process receives your keys and sends them to the corresponding providers. The app excludes keys from saved records and exports. Use **Clear session** to discard your session data and keys. Session cleanup on disconnect is best effort, and the hosting filesystem is temporary. Download reports you want to keep.

Everything starts from an explicit user action. There are no scheduled jobs or automatic paid calls. The free hosted app can sleep when idle and may need time to wake.

## Run locally

Use Python 3.12 or later.

```sh
git clone https://github.com/jessepcc/london-office-market-demo.git
cd london-office-market-demo
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m streamlit run app.py
```

Open the URL printed by Streamlit. Synthetic data loads when the app starts. Enter provider keys in the sidebar for live work. The browser app deliberately ignores provider keys and database paths in the host environment.

For a persistent local command-line database, use the CLI.

```sh
.venv/bin/python -m london_market init-fixtures --db data/demo.sqlite3
.venv/bin/python -m london_market research "What does the evidence show about City office vacancy?" --mode fixture --db data/demo.sqlite3
.venv/bin/python -m london_market briefing --mode fixture --db data/demo.sqlite3
```

CLI live mode accepts `OPENROUTER_API_KEY` and `TAVILY_API_KEY` from your local environment. Keep them out of Git. See [hosting and configuration](docs/hosting.md).

## How it works

Streamlit calls one bounded Python agent runner. The model loads registered runtime skills and chooses approved typed tools for search, retrieval, evidence queries, comparisons, briefing authoring and visualization. Deterministic Python handles validation, arithmetic, deduplication and SQLite storage.

The runtime has 14 registered skills. Chart selection and data preparation are separate responsibilities. Claims retain evidence references. Publication dates, reference periods and retrieval dates remain distinct.

## Demo boundaries

- This is a research PoC for landlords and asset managers. Outputs need human review and do not provide asset valuations or portfolio advice.
- Whole London is the research remit. Available public evidence is uneven, including gaps in Canary Wharf and Midtown measurements and selected samples in wider London.
- Sources may be blocked, stale or inconsistent. No new evidence does not establish that the market is unchanged.
- Charts need compatible observations. Missing history produces a table or an explicit data gap.
- Live research can stop with partial results. Reliable unattended delivery is not a demonstrated capability.
- Saved reports and history belong to the current session. Report downloads are standalone HTML.

The presentation includes dated recordings of an earlier interface. Its historic report and timings are examples, not a current market feed or a performance guarantee.

## Development

```sh
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

See [demo scope](docs/demo-scope.md) and [verification](docs/verification.md).
