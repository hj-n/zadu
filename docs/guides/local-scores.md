# Pointwise scores

Some measures expose per-sample distortions in addition to their global score.
Set `return_local=True` when constructing the runner:

The example uses the arrays from the [quickstart](../getting-started/quickstart.md).

```python
from zadu import ZADU

specs = [
    {"id": "dtm", "params": {}},
    {"id": "mrre", "params": {"k": 30}},
]

runner = ZADU(specs, original, return_local=True)
global_scores, local_scores = runner.measure(projection)

print(global_scores[1])
print(local_scores[1]["local_mrre_false"])
print(local_scores[1]["local_mrre_missing"])
```

Both lists follow specification order. A measure that does not support local
output contributes `None` at its position in `local_scores`.

## Supported measures

The following measures provide pointwise scores:

- Trustworthiness & Continuity (`tnc`)
- Mean Relative Rank Error (`mrre`)
- Local Continuity Meta-Criteria (`lcmc`)
- Neighborhood Hit (`nh`)
- Class-Aware Trustworthiness & Continuity (`ca_tnc`)
- Steadiness & Cohesiveness (`snc`)

Each local array has shape `(n,)` in sample row order. The direction matches
the corresponding global output: larger values mean better preservation for
the six supported measures. Preserve row order when joining values to sample
identifiers or plotting them. Inspect low-scoring points even when the global
average is high.

Continue to [Visualization](visualization.md) to render paired local
distortions.
