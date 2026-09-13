# Measure reference

ZADU exposes 22 measures. Short aliases and full module IDs are accepted by
`ZADU`; the tables below use the short aliases intended for specifications.
Ranges describe ZADU's outputs. “Best” gives the target value for the
property being measured, not a guarantee that the whole projection is faithful.
`n` is the sample count. Parameters and input restrictions follow the tables.

`nh`, `dsc`, `ivm`, and `c_evm` evaluate a projection against class labels.
`ca_tnc`, `l_tnc`, and `cadi` compare class-related structure between spaces.
All seven require a label for each row.

## Local measures

| Measure and paper | ID | Return keys | Range; best |
| --- | --- | --- | --- |
| Trustworthiness & Continuity<br>[Venna & Kaski (2006)](https://doi.org/10.1016/j.neunet.2006.05.014) | `tnc` | `trustworthiness`, `continuity` | [0, 1]; 1 |
| Mean Relative Rank Error<br>[Lee & Verleysen (2009)](https://doi.org/10.1016/j.neucom.2008.12.017) | `mrre` | `mrre_false`, `mrre_missing` | [0, 1]; 1 |
| Local Continuity Meta-Criteria<br>[Chen & Buja (2009)](https://doi.org/10.1198/jasa.2009.0111) | `lcmc` | `lcmc` | [-k/(n-1), 1-k/(n-1)]; 1-k/(n-1) |
| Neighborhood Hit<br>[Paulovich et al. (2008)](https://doi.org/10.1109/TVCG.2007.70443) | `nh` | `neighborhood_hit` | [0, 1]; 1 |
| Neighbor Dissimilarity<br>[Fujiwara et al. (2023)](https://doi.org/10.1109/PacificVis56936.2023.00021) | `nd` | `neighbor_dissimilarity` | non-negative; 0 |
| Class-Aware Trustworthiness & Continuity<br>[Colange et al. (2020)](https://proceedings.neurips.cc/paper/2020/hash/99607461cdb9c26e2bd5f31b12dcf27a-Abstract.html) | `ca_tnc` | `ca_trustworthiness`, `ca_continuity` | [0, 1]; 1 |
| Procrustes Measure<br>[Goldberg & Ritov (2009)](https://doi.org/10.1007/s10994-009-5107-9) | `proc` | `procrustes` | non-negative; 0 |

## Cluster-level measures

| Measure and paper | ID | Return keys | Range; best |
| --- | --- | --- | --- |
| Steadiness & Cohesiveness<br>[Jeon et al. (2021)](https://doi.org/10.1109/TVCG.2021.3114833) | `snc` | `steadiness`, `cohesiveness` | [0, 1]; 1 |
| Distance Consistency<br>[Sips et al. (2009)](https://doi.org/10.1111/j.1467-8659.2009.01467.x) | `dsc` | `distance_consistency` | [0, 1]; 1 |
| Internal Validation Measure<br>[Silhouette](https://doi.org/10.1016/0377-0427%2887%2990125-7), [Calinski-Harabasz](https://doi.org/10.1080/03610927408827101), [Davies-Bouldin](https://doi.org/10.1109/TPAMI.1979.4766909) | `ivm` | selected measure name | depends on selection; depends on selection |
| Clustering + External Validation<br>[Adjusted Rand](https://doi.org/10.1007/BF01908075), [AMI/NMI](https://www.jmlr.org/papers/v11/vinh10a.html), [V-measure](https://aclanthology.org/D07-1043/) | `c_evm` | `{clustering}_{measure}` | depends on selection; depends on selection |
| Label Trustworthiness & Continuity[^label-tnc]<br>[Jeon et al. (2024)](https://doi.org/10.1109/TVCG.2023.3327187) | `l_tnc` | `label_trustworthiness`, `label_continuity` | [0, 1]; 1 |
| Class Angular Distortion Index<br>[Gunaratne et al. (2026)](https://doi.org/10.1111/cgf.70465) | `cadi` | `class_angular_distortion_index` | [0, 1]; 0 |

[^label-tnc]: ZADU uses its normalized DSC directly in label T&C; it does not
    apply the paper's DSC rescaling step.

## Global measures

| Measure and paper | ID | Return keys | Range; best |
| --- | --- | --- | --- |
| Stress<br>[Kruskal (1964a)](https://doi.org/10.1007/BF02289565), [(1964b)](https://doi.org/10.1007/BF02289694) | `stress` | `stress` | non-negative; 0 |
| Non-Metric Stress<br>[Kruskal (1964)](https://doi.org/10.1007/BF02289565) | `nm_stress` | `non_metric_stress` | non-negative; 0 |
| Scale-Normalized Stress<br>[Smelser et al. (2024)](https://arxiv.org/abs/2408.07724) | `sn_stress` | `scale_normalized_stress` | non-negative; 0 |
| Kullback-Leibler Divergence<br>[Hinton & Roweis (2002)](https://papers.nips.cc/paper/2276-stochastic-neighbor-embedding) | `kl_div` | `kl_divergence` | non-negative; 0 |
| Distance-to-Measure<br>[Chazal et al. (2011)](https://doi.org/10.1007/s10208-011-9098-0) | `dtm` | `distance_to_measure` | non-negative; 0 |
| Topographic Product<br>[Bauer & Pawelzik (1992)](https://doi.org/10.1109/72.143371) | `topo` | `topographic_product` | real-valued; 0 |
| Pearson correlation<br>[Pearson (1895)](https://doi.org/10.1098/rspl.1895.0041) | `pr` | `pearson_r` | [-1, 1]; 1 |
| Spearman rank correlation<br>[Spearman (1904)](https://doi.org/10.2307/1412159) | `srho` | `spearman_rho` | [-1, 1]; 1 |

Pearson and Spearman use each unique off-diagonal distance once. Their distance
vectors must have nonzero variance.

## Gap-based regional measure

| Measure and paper | ID | Return keys | Range; best |
| --- | --- | --- | --- |
| Gap Index[^gap]<br>[Ros et al. (2026)](https://arxiv.org/abs/2607.28324) | `gi` | `gap_index` | [0, 1]; 0 |

Gap Index measures distortion in empty triangular regions of a two-dimensional
projection. See the [dedicated guide](gap-index.md) for regional output.

[^gap]: Introduced by Jaume Ros, Alessio Arleo, and Fernando Paulovich. ZADU's
    adaptation retains the upstream MIT license and pinned provenance; see the
    [third-party notice](https://github.com/hj-n/zadu/blob/master/THIRD_PARTY_NOTICES.md).

## Parameters and input restrictions

### Neighborhood size

`tnc`, `mrre`, `lcmc`, `nh`, `ca_tnc`, `nd`, `topo`, and `proc` default to
`k=20`. Set an integer with `1 <= k < n`; T&C and class-aware T&C require
`k < n / 2`. Thus their default `k=20` needs at least 41 samples.

ZADU returns MRRE as **one minus normalized rank error**, so larger values are
better. LCMC's upper bound is `1 - k / (n - 1)`, not 1. Topographic Product is
signed; closeness to zero, rather than a larger raw value, indicates its target.

### Density bandwidth

`dtm` and `kl_div` use `sigma=0.1`, which must be positive. Each space's distances
are divided by their maximum before applying the kernel
`exp(-(distance / maximum_distance)**2 / sigma)`. Choose `sigma` for the
density scale you want to compare, and keep it fixed across projections.

### Steadiness and Cohesiveness

| Parameter | Default | Meaning |
| --- | --- | --- |
| `iteration` | `150` | Number of sampled iterations |
| `walk_num_ratio` | `0.3` | Walk count relative to the number of samples |
| `alpha` | `0.1` | Positive offset in the similarity-to-distance conversion `1 / (similarity + alpha)` |
| `k` | omitted | Uses `floor(sqrt(n))`; pass an integer to override |
| `clustering_strategy` | `"dbscan"` | Cluster extraction strategy; also accepts `"kmeans"` and `"N-means"` for an integer N |
| `random_state` | `None` | Seed or NumPy generator |
| `n_jobs` | `1` | Iteration workers; collection execution may reduce this to avoid nested ZADU workers |

### CADI

`n_triplets=0` selects `10 * n` sampled triplets; a positive integer selects an
explicit count. `random_seed=None` uses a new random generator. Set an integer
seed for repeatable samples. At least two classes are required, and one class
must contain at least two samples.

### Gap Index

`metric="euclidean"` measures original-space triangle edges. See the
[Gap Index guide](gap-index.md) for other distance functions, precomputed
distances, and regional output.

### Degenerate inputs

Requirements depend on the measure. T&C can handle duplicate points using
stable index ties; that does not make duplicate or constant inputs valid for
every other measure. Pearson and Spearman require nonconstant distance values.
Stress requires nonzero original distances; scale-normalized stress,
non-metric stress, and density measures require nonzero distances in both spaces.
Procrustes requires
nonzero variance in each original neighborhood; Topographic Product requires
positive distances in its ratios. IVM requires between 2 and `n - 1` classes.

## String-valued options

### Internal validation

`ivm` accepts `silhouette`, `calinski_harabasz`, or `davies_bouldin` as its
`measure`. Silhouette ranges from −1 to 1 (higher is better),
Calinski–Harabasz is non-negative (higher is better), and Davies–Bouldin is
non-negative (lower is better).

### Clustering and external validation

`c_evm` accepts:

- `measure`: `arand`, `ami`, `nmi`, or `vmeasure`
- `clustering`: `kmeans` or `dbscan`

With `clustering="kmeans"`, `n_clusters` defaults to the number of unique
labels and `random_state` defaults to `0`. Override them through
`clustering_args`. All four validation scores are higher-is-better, with 1
indicating agreement. Adjusted Rand and adjusted mutual information may be
negative; normalized mutual information and V-measure range from 0 to 1.
DBSCAN's noise label is passed to the selected scorer along with cluster labels.

### Label trustworthiness and continuity

`l_tnc` accepts `dsc` or `ch_btw` as its `cvm`.

Invalid option strings raise `ValueError` with the allowed values.

## Pointwise return keys

With `return_local=True`, the runner returns `(global_scores, local_scores)`.
Both lists follow specification order; unsupported local entries are `None`.
The local dictionaries use the keys below. Each value is an array of length `n`.

| ID | Local return keys |
| --- | --- |
| `tnc` | `local_trustworthiness`, `local_continuity` |
| `mrre` | `local_mrre_false`, `local_mrre_missing` |
| `lcmc` | `local_lcmc` |
| `nh` | `local_neighborhood_hit` |
| `ca_tnc` | `local_ca_trustworthiness`, `local_ca_continuity` |
| `snc` | `local_steadiness`, `local_cohesiveness` |

The primary references for the original 17 ZADU measures follow Table 1 of the
[ZADU paper](https://doi.org/10.1109/VIS54172.2023.00048). Later additions link
to the publications that introduced them.
