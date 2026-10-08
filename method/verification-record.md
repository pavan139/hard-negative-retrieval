# Retriever verification record

Current test counts, source digest, environment, and stage outcomes are in [reports/verification.json](../reports/verification.json). The report covers numerical and data-boundary tests, command JSON, package metadata, and an installed wheel outside the checkout.

The layout keeps the behavior suite in `tests/` and the entry point at `bin/check_retriever.py`. [Numerical checks](numerical-checks.md) describes the regressions; [retrieval runs](retrieval-runs.md) explains what the benchmark results establish. Model or corpus accuracy should be read from those experiment reports, not inferred from a successful package build.
