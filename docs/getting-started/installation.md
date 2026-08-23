# Installation

ZADU requires Python 3.10 or newer.

## Base package

Install the default NumPy/SciPy execution path from PyPI:

```bash
python -m pip install zadu
```

The base package is sufficient for all 22 measures. Optional frameworks are
not imported or installed unless you request them.

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

An installed framework is never selected implicitly. Configure a backend with
`ExecutionConfig`, and benchmark the actual data and metric specification
before assuming it will be faster. The [backend capability table](../backends.md)
documents supported devices, dtypes, fallbacks, and validation status.

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
