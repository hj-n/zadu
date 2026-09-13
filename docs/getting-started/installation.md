# Installation

ZADU requires Python 3.10 or newer.

## Base package

Install the default NumPy/SciPy execution path from PyPI:

```bash
python -m pip install zadu
```

The base package supports all 22 measures and installs NumPy, SciPy,
scikit-learn, Numba, HDBSCAN, and threadpoolctl. Matplotlib, MLX, and PyTorch
are optional.

## Optional visualization

Install Matplotlib and ZADUVis helpers with:

```bash
python -m pip install "zadu[vis]"
```

See [Visualization](../guides/visualization.md) for an example.

## Optional accelerator backends

Apple Silicon users can install MLX separately:

```bash
python -m pip install "zadu[mlx]"
```

Install PyTorch support with:

```bash
python -m pip install "zadu[torch]"
```

Installing an accelerator does not select it. Set `backend`, `device`, and
`dtype` in `ExecutionConfig`; see [Execution backends](../backends.md).

## Development installation

Clone the repository and install an editable development environment:

```bash
git clone https://github.com/hj-n/zadu.git
cd zadu
python -m pip install -e ".[dev]"
```

Documentation dependencies are separate:

```bash
python -m pip install -e ".[docs]"
python -m mkdocs serve
```
