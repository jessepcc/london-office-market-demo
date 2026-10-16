# Public demo verification

Checks executed on 16 October 2026. Synthetic checks establish application behavior with scripted models. They do not establish current market accuracy or live provider reliability.

| Check | Status | Evidence or limitation |
| --- | --- | --- |
| Retained Python suite | PASS | uv sync --frozen --extra test installed a fresh Python 3.12 environment. .venv/bin/python -m pytest -q passed 514 tests. |
| Credential-free app and CLI | PASS | Fixture initialization, manual collection, manual briefing and HTML export executed. Repeat initialization added zero records. Browser chat and manual briefing completed with labeled synthetic data, two rendered charts and visible coverage gaps. |
| Independent visitor data and keys | PASS | tests/test_web_session.py creates two real AppTest sessions and checks distinct databases, conversations and credentials. |
| Credential clearing and no host-key fallback | PASS | The same tests check ignored host credentials, both-key action gating and directory/key removal on Clear session. |
| Unsupported models and navigation warning | PASS | Two regressions failed before the fixes, then passed with supported-model validation and saved-report navigation. |
| Scheduler and passcode removed | PASS | Source review found no runtime scheduling/auth modules, due-slot schema or automatic job path. CLI rejects the removed scheduler command and poll flags. |
| Desktop landing page | PASS | Rendered in Chrome at 1512px. Recorded image assets loaded; local videos do not autoplay. |
| Mobile landing page | PASS scoped | Chrome device preview measured 390 × 844 CSS pixels. Document scroll width was 390px, with no horizontal overflow. Inspected the full-page capture; both videos remained paused without autoplay. This is browser emulation, not a physical-phone check. |
| Public snapshot content and secret scan | PASS scoped | Reviewed source allowlist and new root history. Pattern scan found no provider/GitHub token signatures, private deployment identifiers or original private project paths. This is not an exhaustive security audit. |
| GitHub Pages publication | PASS | Public repository and Pages workflow created. The publication workflow completed successfully; page, CSS, JavaScript, posters and MP4 assets returned HTTP 200. |
| Free interactive hosting | PASS scoped | app.py deployed from public main on Streamlit Community Cloud, Python 3.12, with no shared secrets. Sharing settings showed public and searchable. The canonical address rendered the synthetic app in an incognito visitor window without sign-in or a passcode. Initial hosted synthetic chat passed. |
| Hosting page consistency | FAIL scoped | The canonical address displayed error 1ST, then stayed loading in the signed-in browser session even after one reboot. The same address opened in incognito. The direct app route worked in the signed-in session; a fallback link is published. The cause remains unconfirmed. See docs/hosting.md. |
| Hosted synthetic briefing and export | PASS scoped | Prepare weekly briefing and Open saved briefing worked both through the direct app route and the canonical URL in an incognito visitor window. Two charts rendered with explicit synthetic labels and coverage gaps. The direct-route download produced standalone HTML with inline Plotly and source references. No widget-default warning, stale Streamlit elements or application exception were present in the checked direct-route state. This does not resolve the signed-in hosting-page loading issue. |
| Live paid research after conversion | NOT RUN | No paid provider calls used for migration checks. |

The public suite retains reusable application and source-fixture checks. It excludes old passcode/scheduler checks, checks for private developer scripts, and 67 test functions replaying omitted private working artifacts. Manual overlap and interruption-recovery behavior is retained in tests/test_jobs.py.

Independent review found and rechecked corrections to cost wording and supported model selection. It found no operator-key fallback or cross-session database sharing. The old paid deployment was removed after a private SQLite backup passed integrity checking. No cloud database or deployment credentials are in this repository.
