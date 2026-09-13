# Direct measure functions

Each module under `zadu.measures` exposes a `measure()` function. Signatures
differ: some require both spaces, while others require only a projection and
labels. This example uses the arrays from the
[quickstart](../getting-started/quickstart.md):

```python
from zadu.measures import mean_relative_rank_error, neighborhood_hit, pearson_r

mrre = mean_relative_rank_error.measure(original, projection, k=20)
pearson = pearson_r.measure(original, projection)
hit = neighborhood_hit.measure(projection, labels, k=20)
```

For the same parameters and distance definitions, direct calls compute the
same scores as the runner. Functions with `return_local=True` return a
`(global_dict, local_dict)` tuple. Other functions return a score dictionary;
Gap Index also offers a [regional API](../measures/gap-index.md).

Some functions accept precomputed distances or neighbors. They must describe
the same samples, row order, distance definition, and neighbor convention as
the calculation. Prefer the runner when combining measures so these resources
are shared consistently. Typed reduction arguments such as `pair_statistics`
and `rank_comparisons` are engine interfaces and normally remain unset.

A direct call does not use a runner's memory budget. It can allocate dense
distance or rank matrices even if the equivalent scheduled calculation uses
blocks. See [Memory and execution](../guides/execution.md).

## Module mapping

| Alias | Module |
| --- | --- |
| `tnc` | `trustworthiness_continuity` |
| `mrre` | `mean_relative_rank_error` |
| `lcmc` | `local_continuity_meta_criteria` |
| `nh` | `neighborhood_hit` |
| `ca_tnc` | `class_aware_trustworthiness_continuity` |
| `l_tnc` | `label_trustworthiness_and_continuity` |
| `nd` | `neighbor_dissimilarity` |
| `dtm` | `distance_to_measure` |
| `kl_div` | `kl_divergence` |
| `dsc` | `distance_consistency` |
| `pr` | `pearson_r` |
| `srho` | `spearman_rho` |
| `ivm` | `internal_validation_measure` |
| `c_evm` | `clustering_and_external_validation_measure` |
| `snc` | `steadiness_cohesiveness` |
| `topo` | `topographic_product` |
| `proc` | `procrustes` |
| `stress` | `stress` |
| `sn_stress` | `scale_normalized_stress` |
| `nm_stress` | `non_metric_stress` |
| `cadi` | `class_angular_distortion_index` |
| `gi` | `gap_index` |

Import modules explicitly in reusable code instead of using a wildcard import.
Parameters and return keys are collected in the
[measure reference](../measures/index.md).
