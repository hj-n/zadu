<p align="center">
  <h2 align="center">ZADU</h2>
  <p align="center"><b>A</b>-to-<b>Z</b> Python library for eval<b>U</b>ating <b>D</b>imensionality reduction</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/zadu/"><img alt="PyPI" src="https://img.shields.io/pypi/v/zadu"></a>
  <a href="https://github.com/hj-n/zadu/actions/workflows/test.yml"><img alt="Tests" src="https://github.com/hj-n/zadu/actions/workflows/test.yml/badge.svg"></a>
  <a href="https://hyeonword.com/zadu/"><img alt="Documentation" src="https://img.shields.io/badge/docs-GitHub%20Pages-4051b5"></a>
</p>

ZADU evaluates how faithfully a dimensionality-reduction projection preserves
its original data. It provides 22 local, cluster-level, global, and gap-based
distortion measures through one consistent Python interface, with exact shared
execution, bounded-memory strategies, repeated-projection evaluation, and
optional MLX and PyTorch backends.

**[Documentation](https://hyeonword.com/zadu/)** ·
**[Measure reference](https://hyeonword.com/zadu/measures/)** ·
**[Performance report](https://hyeonword.com/zadu/performance/0.5.1-acceleration-report/)** ·
**[ZADU paper](https://doi.org/10.1109/VIS54172.2023.00048)**

## Installation

```bash
python -m pip install zadu
```

Visualization is optional:

```bash
python -m pip install "zadu[vis]"
```

See the [installation guide](https://hyeonword.com/zadu/getting-started/installation/)
for optional MLX and PyTorch backends.

## Quick start

```python
import numpy as np
from zadu import ZADU

rng = np.random.default_rng(0)
original = rng.normal(size=(200, 16))
projection = original[:, :2] + 0.05 * rng.normal(size=(200, 2))

specs = [
    {"id": "tnc", "params": {"k": 20}},
    {"id": "mrre", "params": {"k": 20}},
]

scores = ZADU(specs, original).measure(projection)
print(scores)
```

ZADU's execution DAG shares compatible exact distances, neighbors, ranks,
densities, and pair reductions across measures. Scientific scores remain
separate from backend, timing, and memory diagnostics in `last_run_info`.

Read the [quickstart](https://hyeonword.com/zadu/getting-started/quickstart/),
[choose measures](https://hyeonword.com/zadu/guides/choosing-measures/), or
browse the complete [measure reference](https://hyeonword.com/zadu/measures/).

## Contributing

To propose a distortion measure, provide its name, paper, and an optional
reference implementation through the
[metric proposal form](https://github.com/hj-n/zadu/issues/new?template=metric-proposal.yml).
You do not need to learn ZADU's internals or prepare repository files. See
[CONTRIBUTING.md](CONTRIBUTING.md) for development and correctness requirements.

## Citation

```bibtex
@INPROCEEDINGS{jeon23vis,
  author={Jeon, Hyeon and Cho, Aeri and Jang, Jinhwa and Lee, Soohyun and Hyun, Jake and Ko, Hyung-Kwon and Jo, Jaemin and Seo, Jinwook},
  booktitle={2023 IEEE Visualization and Visual Analytics (VIS)},
  title={ZADU: A Python Library for Evaluating the Reliability of Dimensionality Reduction Embeddings},
  year={2023},
  pages={196--200},
  doi={10.1109/VIS54172.2023.00048}
}
```

Each metric's original literature and later additions are credited in the
[measure reference](https://hyeonword.com/zadu/measures/).
