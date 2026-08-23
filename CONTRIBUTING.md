# Contributing to ZADU

Bug fixes, documentation improvements, performance work, and new distortion
measures are welcome.

## Propose a metric without writing code

You do not need to understand ZADU's internals or prepare a specification file.
Open a [GitHub issue](https://github.com/hj-n/zadu/issues/new) and provide:

- the paper or another authoritative description;
- the reference implementation, if one exists; and
- any context you want the maintainers to know.

For example:

```text
Please add the Gap Index.

Paper: https://arxiv.org/abs/2607.28324
Reference implementation: https://codeberg.org/jros/gap-index
```

That is enough for a coding agent or maintainer to investigate the formula,
license, attribution, validation rules, integration, tests, documentation, and
whether an existing accelerated execution resource can be reused. Contributors
do not need to understand the execution planner. Questions should be limited to
scientific or licensing ambiguities that the provided sources cannot resolve.

Every metric contribution must include evidence of correctness: an independent
slow oracle, results pinned to a reference implementation revision, or
analytical examples. Original authors and compatible upstream licenses must be
credited, and adaptations must be described.

## Development setup

Use Python 3.10 or newer and install the development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Before opening a pull request, run:

```bash
python -m pytest
ruff check .
black --check src test scalability_eval benchmarks
```

Coding agents should also follow [AGENTS.md](AGENTS.md), which contains the
repository-specific metric integration checklist. Pull requests should explain
the user-visible change, the verification performed, and any compatibility,
performance, numerical, citation, or licensing implications.
