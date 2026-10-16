# Public demo scope

The public demo supports independent London office research, manual source collection, manual briefing generation, report history, evidence-linked charts and source inspection. It runs through the existing bounded Python agent and registered runtime skills.

The public version removes automatic collection, automatic weekly delivery, a shared access passcode and operator-funded API access. Manual briefings retain their dates and filing weeks. This change does not establish reliable autonomous report completion.

The website adapts the original six-part business presentation into a visual landing page. It explains the business purpose, research process, recorded report and chat examples, next steps, runtime skills, sources and quality limits. It does not publish internal review transcripts or databases.

## Delivery decisions

| Choice | Reason |
| --- | --- |
| GitHub Pages for the presentation | Static hosting with no application server or build dependencies. |
| Streamlit Community Cloud for the interactive demo | Reuses the Python evidence tools and interface on free hosting. |
| Session-owned temporary databases | Keeps visitors' research, traces and reports separate. |
| Visitor-supplied provider keys | Prevents public visitors from spending an operator's shared API budget. |
| Manual actions only | No idle research work or recurring model/search costs. |
| Clean public source snapshot | Keeps private history, credentials, deployment metadata and internal working artifacts out of the public repository. |
| Locally hosted screenshots and MP4 recordings | No third-party analytics, embeds or presentation service. Videos load only when requested. |

## Evidence rules

Preserve publication, event, reference-period, retrieval and briefing dates. Distinguish vacancy from availability, asking rents from achieved rents, refurbishment from new construction and pipeline from available space. Compare observations only when definitions, publishers, units, geographies and periods support the comparison.

Separate facts, interpretation, forecasts and watchpoints. Unknown values remain unknown. Show missing history, failed sources and geography gaps. Newly discovered chat sources do not automatically become normalized feeds.

## Cost and persistence

GitHub Pages and Streamlit Community Cloud provide the hosting for this demo. They have usage limits and no promise of continuous availability. OpenRouter and Tavily usage is billed under each visitor's provider account. App run limits constrain work, but provider billing is authoritative.

Web data is ephemeral and private to one session. The CLI can keep local SQLite history. No database or key is included in the public source snapshot.
