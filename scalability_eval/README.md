# Scalability benchmark

This directory contains the historical scheduler benchmark. Install its optional dependencies with:

```bash
pip install -e ".[benchmark]"
```

The original benchmark datasets are intentionally not committed. Place each compressed dataset below `scalability_eval/data/compressed/<dataset-name>/` using the format expected by `data/reader.py`, then run `python scalability_eval/eval.py` from the repository root.

The benchmark times UMAP optimization together with metric evaluation and should not be interpreted as an isolated per-metric benchmark. The checked-in CSV files are historical results; record the hardware, Python version, dependency versions, data revision, and random seeds for new runs.
