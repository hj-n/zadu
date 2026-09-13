# Choose measures

Choose a measure for the structure you want to preserve. Two projections can
have similar neighborhood scores and very different distance or class scores.

## Start with the question

| Question | Measures | How to read them |
| --- | --- | --- |
| Which original neighbors are lost, and which new neighbors appear? | `tnc`, `mrre` | Paired scores separate false and missing neighbors; higher is better in ZADU |
| How much do the two neighbor sets overlap? | `lcmc` | Overlap corrected for expected random overlap; the upper bound depends on `k` and `n` |
| Do numerical distances match? | `stress` | Lower is better; sensitive to overall scale |
| Do distances match after a global rescaling? | `sn_stress` | Lower is better; fits one scale factor before measuring residuals |
| Are distances linearly or monotonically related? | `pr`, `srho`, `nm_stress` | Correlations are higher-is-better; non-metric stress is lower-is-better |
| Are samples with the same label close together? | `nh`, `dsc`, `ivm`, `c_evm` | Evaluate the projection against labels; do not by themselves establish preservation of original geometry |
| How does class structure change between spaces? | `ca_tnc`, `l_tnc`, `cadi` | Compare class-related neighborhoods, separation, or angles |
| Are groups split or merged without class labels? | `snc` | Steadiness and Cohesiveness use sampled walks; set a seed for comparisons |
| Does the distribution of local density change? | `dtm`, `kl_div` | Compare density estimates; bandwidth `sigma` affects the result |
| Does local geometry change? | `topo`, `proc`, `nd` | Compare neighbor ordering, local alignment, or dissimilarity |
| Do empty scatterplot regions change relative area? | `gi` | Compares triangles from the 2D projection; lower is better |

The [measure reference](../measures/index.md) lists parameters, return keys,
ranges, and papers. ZADU's MRRE outputs are normalized preservation scores:
**1 is best**, despite “error” in the metric's name.

## Compare neighborhoods and distances

For a first comparison, pair T&C with a distance measure. Using `original` and
`projection` from the [quickstart](../getting-started/quickstart.md):

```python
from zadu import ZADU

specs = [
    {"id": "tnc", "params": {"k": 10}},
    {"id": "sn_stress"},
]
scores = ZADU(specs, original).measure(projection)
```

Use `stress` instead of `sn_stress` if absolute distance scale matters. Add
`gi` when empty regions in a two-dimensional scatterplot are part of your
question. Add class-based measures when class structure is relevant.

## Interpret a comparison

Keep sample selection, preprocessing, and measure parameters fixed when
comparing projections. Changing `k` changes the neighborhood scale being
evaluated. Try several values if conclusions depend on the size of a
neighborhood, and report each value with its result.

A label score can be high even when the projection separates classes that
overlap in the original space. Use a measure that compares both spaces when
making a claim about preservation. Scores from different measures have
different meanings and units; report them separately.

For reproducibility, record the ZADU version, measure parameters, preprocessing,
sample selection, and backend/device/dtype. Set `random_state` for S&C and
`random_seed` for CADI. A single execution worker does not make an unseeded
randomized measure deterministic.
