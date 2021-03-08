## Quantitative metrics of Multidimensional Projections (MDP) in Python

**Based on the code written by [Hyung-Kwon Ko](https://github.com/hyungkwonko/umato)**

### Installation

```
pip3 install metrics4mdp
```

### Usage


#### Example
```python
from metric4mdp import MDPMetricProvider

mlist = [""DTM", "DTM_KL1", "Continuity", "Trustworthiness"]
metricprovider = MDPMetricProvider(raw_data, emb_data, mlist, k=7)


```