# Visualization

ZADUVis is an optional layer for rendering pointwise distortion scores with
CheckViz and Reliability Map. It is kept out of the base installation because
scientific evaluation does not require Matplotlib.

```bash
python -m pip install "zadu[vis]"
```

## Compute and render local scores

```python
import matplotlib.pyplot as plt
from zadu import ZADU
from zaduvis import zaduvis

specs = [{"id": "tnc", "params": {"k": 25}}]
runner = ZADU(specs, original, return_local=True)
_, local_scores = runner.measure(projection)

local = local_scores[0]
trustworthiness = local["local_trustworthiness"]
continuity = local["local_continuity"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
zaduvis.checkviz(
    projection,
    trustworthiness,
    continuity,
    ax=axes[0],
)
zaduvis.reliability_map(
    projection,
    trustworthiness,
    continuity,
    k=10,
    ax=axes[1],
)
plt.show()
```

![ZADUVis example](https://github.com/hj-n/zadu/assets/37105201/7c6dc8d7-59c5-48fd-92a5-186e1e44597a)

CheckViz originates from
[Lespinats and Aupetit (2011)](https://doi.org/10.1111/j.1467-8659.2010.01835.x).
Reliability Map is described by
[Jeon et al. (2022)](https://doi.org/10.48550/arXiv.2107.07859).
