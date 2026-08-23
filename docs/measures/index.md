---
hide:
  - toc
---

# Measure reference

ZADU exposes 22 measures. Short aliases and full module IDs are accepted by
`ZADU`; the tables below use the short aliases intended for specifications.
Ranges describe the implemented definition, and “optimum” indicates the
direction of perfect preservation when one exists.

!!! warning "Class-label assumptions"

    `dsc`, `ivm`, `c_evm`, `nh`, and `ca_tnc` assess a projection relative to
    labels that are assumed to describe meaningful, sufficiently separated
    original-space structure. Interpret them cautiously when that assumption
    is not supported.

## Local measures

| Measure | ID | Parameters | Return keys | Range | Optimum | Primary reference |
| --- | --- | --- | --- | --- | --- | --- |
| Trustworthiness & Continuity | `tnc` | `k=20` | `trustworthiness`, `continuity` | [0.5, 1] | 1 | [Venna & Kaski (2006)](https://doi.org/10.1016/j.neunet.2006.05.014) |
| Mean Relative Rank Error | `mrre` | `k=20` | `mrre_false`, `mrre_missing` | [0, 1] | 1 | [Lee & Verleysen (2009)](https://doi.org/10.1016/j.neucom.2008.12.017) |
| Local Continuity Meta-Criteria | `lcmc` | `k=20` | `lcmc` | [-k/(n-1), 1-k/(n-1)] | 1-k/(n-1) | [Chen & Buja (2009)](https://doi.org/10.1198/jasa.2009.0111) |
| Neighborhood Hit | `nh` | `k=20` | `neighborhood_hit` | [0, 1] | 1 | [Paulovich et al. (2008)](https://doi.org/10.1109/TVCG.2007.70443) |
| Neighbor Dissimilarity | `nd` | `k=20` | `neighbor_dissimilarity` | non-negative | 0 | [Fujiwara et al. (2023)](https://doi.org/10.1109/PacificVis56936.2023.00021) |
| Class-Aware Trustworthiness & Continuity | `ca_tnc` | `k=20` | `ca_trustworthiness`, `ca_continuity` | [0.5, 1] | 1 | [Colange et al. (2020)](https://proceedings.neurips.cc/paper/2020/hash/99607461cdb9c26e2bd5f31b12dcf27a-Abstract.html) |
| Procrustes Measure | `proc` | `k=20` | `procrustes` | non-negative | 0 | [Goldberg & Ritov (2009)](https://doi.org/10.1007/s10994-009-5107-9) |

## Cluster-level measures

| Measure | ID | Parameters | Return keys | Range | Optimum | Primary reference |
| --- | --- | --- | --- | --- | --- | --- |
| Steadiness & Cohesiveness | `snc` | `iteration=150`, `walk_num_ratio=0.3`, `alpha=0.1`, `k=None`, `clustering_strategy="dbscan"`, `random_state=None`, `n_jobs=1` | `steadiness`, `cohesiveness` | [0, 1] | 1 | [Jeon et al. (2021)](https://doi.org/10.1109/TVCG.2021.3114833) |
| Distance Consistency | `dsc` | none | `distance_consistency` | [0, 1] | 1 | [Sips et al. (2009)](https://doi.org/10.1111/j.1467-8659.2009.01467.x) |
| Internal Validation Measure | `ivm` | `measure="silhouette"` | selected measure name | depends on selection | depends on selection | [Silhouette](https://doi.org/10.1016/0377-0427%2887%2990125-7), [Calinski-Harabasz](https://doi.org/10.1080/03610927408827101), [Davies-Bouldin](https://doi.org/10.1109/TPAMI.1979.4766909) |
| Clustering + External Validation | `c_evm` | `measure="arand"`, `clustering="kmeans"`, `clustering_args=None` | `{clustering}_{measure}` | depends on selection | depends on selection | [Adjusted Rand](https://doi.org/10.1007/BF01908075), [AMI/NMI](https://www.jmlr.org/papers/v11/vinh10a.html), [V-measure](https://aclanthology.org/D07-1043/) |
| Label Trustworthiness & Continuity[^label-tnc] | `l_tnc` | `cvm="dsc"` | `label_trustworthiness`, `label_continuity` | [0, 1] | 1 | [Jeon et al. (2024)](https://doi.org/10.1109/TVCG.2023.3327187) |
| Class Angular Distortion Index | `cadi` | `n_triplets=0`, `random_seed=None` | `class_angular_distortion_index` | [0, 1] | 0 | [Gunaratne et al. (2026)](https://doi.org/10.1111/cgf.70465) |

[^label-tnc]: The implementation does not apply the original paper's DSC
    rescaling step. The transformation was intended to map DSC into [0, 1],
    which is unnecessary for ZADU's already normalized DSC score.

## Global measures

| Measure | ID | Parameters | Return key | Range | Optimum | Primary reference |
| --- | --- | --- | --- | --- | --- | --- |
| Stress | `stress` | none | `stress` | non-negative | 0 | [Kruskal (1964a)](https://doi.org/10.1007/BF02289565), [(1964b)](https://doi.org/10.1007/BF02289694) |
| Non-Metric Stress | `nm_stress` | none | `non_metric_stress` | non-negative | 0 | [Kruskal (1964)](https://doi.org/10.1007/BF02289565) |
| Scale-Normalized Stress | `sn_stress` | none | `scale_normalized_stress` | non-negative | 0 | [Smelser et al. (2024)](https://arxiv.org/abs/2408.07724) |
| Kullback-Leibler Divergence | `kl_div` | `sigma=0.1` | `kl_divergence` | non-negative | 0 | [Hinton & Roweis (2002)](https://papers.nips.cc/paper/2276-stochastic-neighbor-embedding) |
| Distance-to-Measure | `dtm` | `sigma=0.1` | `distance_to_measure` | non-negative | 0 | [Chazal et al. (2011)](https://doi.org/10.1007/s10208-011-9098-0) |
| Topographic Product | `topo` | `k=20` | `topographic_product` | real-valued | 0 | [Bauer & Pawelzik (1992)](https://doi.org/10.1109/72.143371) |
| Pearson correlation | `pr` | none | `pearson_r` | [-1, 1] | 1 | [Pearson (1895)](https://doi.org/10.1098/rspl.1895.0041) |
| Spearman rank correlation | `srho` | none | `spearman_rho` | [-1, 1] | 1 | [Spearman (1904)](https://doi.org/10.2307/1412159) |

Pearson and Spearman use each unique off-diagonal distance once. The stress
and density families reject all-zero distance inputs because their
normalizations are undefined.

## Gap-based regional measure

| Measure | ID | Parameters | Return key | Range | Optimum | Primary reference |
| --- | --- | --- | --- | --- | --- | --- |
| Gap Index[^gap] | `gi` | `metric="euclidean"` | `gap_index` | [0, 1] | 0 | [Ros et al. (2026)](https://arxiv.org/abs/2607.28324) |

Gap Index measures distortion in empty triangular regions of a two-dimensional
projection. It does not fit cleanly into the local, cluster, or global families.
See the [dedicated guide](gap-index.md) for detailed and regional output.

[^gap]: Introduced by Jaume Ros, Alessio Arleo, and Fernando Paulovich. ZADU's
    adaptation retains the upstream MIT license and pinned provenance; see the
    [third-party notice](https://github.com/hj-n/zadu/blob/master/THIRD_PARTY_NOTICES.md).

## String-valued options

### Internal validation

`ivm` accepts `silhouette`, `calinski_harabasz`, or `davies_bouldin` as its
`measure`.

### Clustering and external validation

`c_evm` accepts:

- `measure`: `arand`, `ami`, `nmi`, or `vmeasure`
- `clustering`: `kmeans` or `dbscan`

With `clustering="kmeans"`, `n_clusters` defaults to the number of unique
labels and `random_state` defaults to `0`. Override them through
`clustering_args`.

### Label trustworthiness and continuity

`l_tnc` accepts `dsc` or `ch_btw` as its `cvm`.

Invalid option strings raise `ValueError` with the allowed values.

## Pointwise return keys

With `return_local=True`, supported measures additionally return:

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
