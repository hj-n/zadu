# Choose measures

There is no universally best projection-quality score. Start from the
scientific structure you need to preserve, then use complementary measures
rather than selecting only the most favorable result.

## Match the question to a measure family

| Question | Useful starting points | What they emphasize |
| --- | --- | --- |
| Are local neighbors preserved? | T&C, MRRE, LCMC | Missing and false neighbors or neighborhood overlap |
| Are labels or classes visually coherent? | Neighborhood Hit, DSC, class-aware T&C, label T&C, CADI | Separation, mixing, and class-relative distortions |
| Are pairwise distances preserved? | Stress, scale-normalized stress, Pearson | Magnitude or linear association of distances |
| Is the ordering of distances preserved? | Spearman, non-metric stress | Monotonic rather than metric agreement |
| Are density patterns preserved? | DTM, KL divergence | Changes in local density estimates |
| Is local topology preserved? | Topographic Product, Procrustes | Neighborhood ordering or locally aligned geometry |
| Do apparent empty regions represent real separation? | Gap Index | Empty triangular regions in a 2D projection |
| Are cluster structures stable without labels? | Steadiness & Cohesiveness | False and missing groups discovered through random walks |

The [measure reference](../measures/index.md) lists exact IDs, parameters,
ranges, optimum directions, return keys, and original literature.

## A practical baseline

For an unlabeled two-dimensional projection, a useful first pass combines one
local, one global, and one structural measure:

```python
specs = [
    {"id": "tnc", "params": {"k": 20}},
    {"id": "stress", "params": {}},
    {"id": "gi", "params": {"metric": "euclidean"}},
]
```

This is not a universal prescription. Change `k`, add scale-aware or
class-aware measures, and validate that the selected definition matches the
claim you intend to make.

## Interpret scores carefully

- Some measures are maximized at 1; others are minimized at 0.
- `k` defines the neighborhood scale. Report it with the score and, when
  possible, examine more than one scientifically meaningful scale.
- Label-based visual-quality measures assume that the labels describe
  meaningful structure in the original space. A visually separated projection
  is not automatically faithful if those labels overlap before projection.
- Scores from different measures do not share a common unit. Do not average
  them without a justified model.
- A faster backend or `float32` changes execution characteristics, not the
  published definition, but floating-point tolerances can differ by dtype.

## Reproducibility checklist

Record the ZADU version, measure IDs, all non-default parameters, preprocessing,
sample selection, backend/device/dtype, and whether local or global scores were
used. For randomized measures such as S&C or sampled CADI, also set and report
the seed.
