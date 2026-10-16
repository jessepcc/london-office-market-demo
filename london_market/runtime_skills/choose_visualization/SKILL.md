---
name: choose_visualization
description: Choose supported trends, comparisons, volumes, composition or descriptive relationships from validated observations.
---

# Choose an evidence-supported graph

Name the business question. Inspect market_inventory series, definitions, periods and source excerpts. Call choose_visualization with validated observation IDs and the matching preferred intent. Python validates suitability. Keep choose and data preparation separate.

- trend: one metric and one unit on the axis, using compatible dated history; aligned submarkets may form multiple lines. Missing or overlapping periods are not interpolated.
- submarkets: the same metric, publisher, definition and period across distinct areas. Central London and its components overlap.
- total: dated source-reported values shown as columns, without inventing an aggregate or implying a trend from one point.
- proportion: directly reported percentage shares with reviewed nonoverlapping segments. Amount-derived shares lack a reviewed denominator contract and remain tables. Use source-native dimension/segment labels. No fabricated Other category.
- relation: explicit x/y observation pairs aligned by geography and period, with independent units and consistent definitions on each axis. A scatter is descriptive and does not establish causality.
- table: preserve sparse or incompatible observations honestly.

Inspect the returned caveats and fallback state. Split mixed metrics into separate trend plans, or construct explicit aligned relation pairs. Even a table fallback must pass through prepare_visualization_data before finish can use it. Never override the contract or inject numeric data. Load prepare_visualization_data if required and pass its plan_id. Only its resulting chart_id may appear in finish. If preparation cannot fit the allowance, omit the unprepared chart and keep the supported text finding.

For full-market reports, inspect unpaginated market_inventory.local_exhibit_candidates before choosing convenient aggregate histories. Prepare a useful compatible native-local exhibit when a conflict-free pair exists. The completion boundary rejects avoidable Central-only selection with exact candidate IDs. Keep useful aggregate context too. Shared measurement, publisher, definition, units and reference periods remain mandatory; different cumulative rent windows cannot form a matched chart. A fallback table is honest when no compatible pair exists, but it does not discharge an available valid comparison. Native City Fringe and selected wider districts retain their sampling limits. Attach each prepared exhibit to a coverage entry in a full-market completion.

Full-market reports include the supported trend, relationship, displayed-period flow sum and proportion purposes listed in Run budget.visual_purposes and market_inventory.visual_purposes. These dispositions use all retained same-mode observations before inventory pagination. Inspect their exact supporting excerpts. Choose each missing candidate early; independent choices can share one model turn, followed by a batch of preparation calls using the returned plan IDs. Select every resulting chart ID in finish.chart_ids. A prepared but omitted chart or a fallback table cannot satisfy an attainable purpose. The native-local exhibit remains required separately unless a suitable selected exhibit serves both questions. Each unchosen candidate needs two tool actions, chosen plans need preparation, and model turns plus finish must leave classification protection intact. The harness preserves unavailable-purpose reasons as limitations; do not invent missing series, denominators or residual shares. Focused answers have no four-purpose obligation.
