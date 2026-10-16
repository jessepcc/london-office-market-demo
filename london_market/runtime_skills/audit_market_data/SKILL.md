---
name: audit_market_data
description: Audit stored London market series for missing history, revisions, units, geography and comparison suitability.
---

# Audit market data before analysis

Reuse inventories already inspected in this run. If a needed series is absent, call market_inventory with a precise metric filter and limit 8 for compact series IDs, values, periods, supporting excerpts and coverage. Inspect total counts and continuation. Use precise query_evidence filters for the important records. Identify missing periods, duplicate/revised values, incompatible definitions and retained historical data. Missing values are not zero. Do not impute, delete outliers or resolve conflicting revisions by preference.

A comparable history has the same publisher, source-native measure, geography, definition, category, unit and cadence. Reference intervals must not overlap. Use compare for exactly two validated observations and retain its absolute/relative-change units. Percentage measurements change in percentage points. Different publisher series remain distinct; a disagreement is useful evidence of methodology differences, not an average to conceal them.

For quality/sector shares inspect dimension, segment and the reviewed partition contract. Never combine overlapping categories or invent a denominator. A total is a source-reported total unless disjoint aggregation is explicitly reviewed. A relationship pairs validated measures on aligned periods and geography, with consistent definitions on each axis. It establishes association only, not causality.

Return useful cited findings, or load the relevant topic/visualization package and continue in the same run. This skill grants inventory, query and comparison tools only. It does not perform regressions, statistical tests, forecasting, source promotion or arbitrary code execution.

Coverage rows describe all retained evidence even after filtering a metric. Inspect native_geographies and scope_note before asserting a missing submarket. Exhibit candidates retain exact observation IDs and still require choose_visualization and prepare_visualization_data.
