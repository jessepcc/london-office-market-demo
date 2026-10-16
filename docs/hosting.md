# Hosting and configuration

## Project presentation

GitHub Actions publishes only site/ to GitHub Pages. Push a site change to main or run the Publish project presentation workflow manually. The page uses HTML, CSS and small local JavaScript. There is no Node build or hosted backend.

## Interactive Python demo

Deploy app.py from main on [Streamlit Community Cloud](https://share.streamlit.io/). Choose Python 3.12 in the deployment settings and the subdomain london-office-market-demo. requirements.txt installs the local Python package and its pinned dependencies.

Do not configure shared OpenRouter or Tavily secrets on the host. Visitors enter their own keys in the app. Streamlit receives these keys over its browser session and the Python process passes them to the corresponding provider. They are excluded from persisted records.

The app opens in synthetic mode and loads labeled examples without provider requests. Live mode requires the visitor's credentials. All work is on demand. The free host can sleep while idle, and filesystem contents can disappear on restart. Reports should be downloaded when needed.

Streamlit hosting requires a signed-in account and authorization to read the public GitHub repository. No paid VM, dedicated disk, reserved IP, cron process or background worker is required.

## Local CLI configuration

The browser ignores host environment credentials and database paths. The CLI reads them for local use.

| Variable | Purpose |
| --- | --- |
| LONDON_MARKET_MODE | fixture or live. |
| LONDON_MARKET_DB | Persistent local SQLite path. |
| OPENROUTER_API_KEY | Your local OpenRouter key for live CLI calls. |
| TAVILY_API_KEY | Your local Tavily key for fresh search. |
| OPENROUTER_MAIN_MODEL | Research model identifier. |
| OPENROUTER_ROUTER_MODEL | Routing model identifier. |
| LONDON_MARKET_MAX_STEPS | Bound tool and model actions. |
| LONDON_MARKET_MAX_SECONDS | Bound run duration. |
| LONDON_MARKET_MAX_TOKENS | Bound token usage. |
| LONDON_MARKET_MAX_SPEND_USD | Bound accounted provider cost. |

Never commit .env, provider keys, SQLite files or Streamlit secrets.
