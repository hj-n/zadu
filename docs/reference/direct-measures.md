# Direct measure functions

Each module under `zadu.measures` exposes a public `measure()` function. Direct
calls are useful for one-off research code and detailed APIs such as Gap Index
regional output. The scheduled `ZADU` interface is preferred when multiple
metrics can share exact resources.

```python
from zadu.measures import mean_relative_rank_error, neighborhood_hit, pearson_r

mrre = mean_relative_rank_error.measure(original, embedding, k=20)
pearson = pearson_r.measure(original, embedding)
hit = neighborhood_hit.measure(embedding, labels, k=20)
```

Direct functions validate public inputs and return the same scalar score
dictionaries as scheduled execution. Arguments representing injected
resources—such as `pair_statistics`, `rank_comparisons`, or `knn_info`—are
internal acceleration hooks. Application code should not construct them.

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
