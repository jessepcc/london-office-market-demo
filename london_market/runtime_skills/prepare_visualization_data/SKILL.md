---
name: prepare_visualization_data
description: Prepare deterministic chart values, paired observations, reported shares and evidence provenance from an approved plan.
---

# Prepare validated chart data

Call prepare_visualization_data with a plan_id created by choose_visualization in this run. Python rechecks stored observations and prepares points, paired x/y values, source-reported percentage shares, units, dimensions, periods, publishers and evidence IDs. Do not inject points, interpolate gaps, fabricate Other, or generate executable chart code.

Inspect payload type and caveats. Relationship axes retain distinct metrics and units; proportions retain their source-defined denominator. Missing data is different from zero. A recent retrieval does not make a historical observation current. Unsupported inputs produce a source-linked table or insufficient-data state.

Use the resulting chart_id in finish.chart_ids. A plan_id is not a chart_id. Section chart_indexes refer to finish.chart_ids order. Explain the supported finding and its material qualification in a concise cited claim. Respect the same run allowance as all other skills.
