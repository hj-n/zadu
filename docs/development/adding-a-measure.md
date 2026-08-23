# Adding a measure

A contributor may provide only a paper link and, when available, a reference
implementation. Maintainers or coding agents handle repository integration.
Do not require contributors to learn ZADU's registry or write metadata files.

## Scientific intake

1. Read the paper and reference implementation as scientific sources.
2. Record the authors, paper, pinned reference revision, license, and every
   adaptation in code, tests, documentation, and notices where applicable.
3. Resolve licensing before copying code. Independently implement the formula
   when upstream licensing is unknown or incompatible.
4. Establish correctness with a slow oracle, pinned upstream golden fixture,
   or analytical examples before optimizing.

If the paper and reference implementation disagree, document the discrepancy
and ask only for the scientific decision that cannot be resolved from those
sources.

## Repository integration

The implementation normally touches:

```text
src/zadu/measures/<metric_id>.py
src/zadu/measures/__init__.py
src/zadu/registry.py
src/zadu/spec.py
test/test_<metric_id>.py
docs/measures/
CHANGELOG.md
```

The public `measure()` function must return a dictionary of finite Python
scalar scores. Validate shapes, parameters, labels, and mathematically
undefined inputs with actionable exceptions. Test the direct function and the
scheduled `ZADU` interface, source parity, edge cases, determinism, and every
declared invariance.

## Use the execution DAG when semantics match

Before computing distances, neighbors, ranks, densities, or pair reductions,
inspect `ResourceRequirement` constants in `src/zadu/engine/resources.py`.
Declare and consume a resource only when its exact semantics match the metric.
Keep a direct-call fallback so standalone use remains available.

Add a mixed-specification test and inspect `last_run_info` to prove that the
resource has multiple consumers. Do not request a dense or globally ordered
resource when the formula needs only a small subset. Add a new typed resource
only when the work is both expensive enough to plan and reusable.

## Required checks

```bash
python -m pytest test/test_metric_contract.py -q
python -m pytest
ruff check .
black --check src test scalability_eval benchmarks
python -m mkdocs build --strict
```

The canonical coding-agent checklist remains in
[`AGENTS.md`](https://github.com/hj-n/zadu/blob/master/AGENTS.md).
