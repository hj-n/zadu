# Pointwise scores

Some measures expose per-sample distortions in addition to their global score.
Set `return_local=True` when constructing the runner:

```python
from zadu import ZADU

specs = [
    {"id": "dtm", "params": {}},
    {"id": "mrre", "params": {"k": 30}},
]

runner = ZADU(specs, original, return_local=True)
global_scores, local_scores = runner.measure(embedding)

print(global_scores[1])
print(local_scores[1]["local_mrre_false"])
print(local_scores[1]["local_mrre_missing"])
```

Both lists follow specification order. A measure that does not support local
output contributes `None` at its position in `local_scores`.

## Supported measures

The registry currently exposes pointwise values for:

- Trustworthiness & Continuity (`tnc`)
- Mean Relative Rank Error (`mrre`)
- Local Continuity Meta-Criteria (`lcmc`)
- Neighborhood Hit (`nh`)
- Class-Aware Trustworthiness & Continuity (`ca_tnc`)
- Steadiness & Cohesiveness (`snc`)

Local arrays describe the contribution associated with each row, not an
independent dataset-level metric. Preserve row order when joining them to
identifiers or plotting them.

Continue to [Visualization](visualization.md) to render paired local
distortions.
