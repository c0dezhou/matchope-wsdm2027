# KuaiRand randomized category-matching evidence

This directory contains the fixed-window KuaiRand-Pure analysis used as the
randomized user-level evidence for MatchOPE. The random-intervention rows are
evaluation data only. The target policy is trained on the preceding standard
log; randomized outcomes are not used to fit or select the target or its
strength.

The action is an official first-level video category. Rare categories are
merged using candidate-pool item counts, and the logger's stochastic sampler
within each category is retained. This is therefore a category-level compound
intervention, not exact-video OPE.

The protocol JSON also records a design-known contrast-overlap audit. All 36
pooled categories have positive candidate-pool mass in each of the 50 shared
contexts, so category-level target, reference, and signed unsupported mass are
zero. This uses the published fixed-pool randomization design and metadata,
not observed zero counts.
The release does not expose row-level exact-video eligibility or propensity,
so the same audit explicitly leaves exact-video support uncertified.

The official category file is sorted by `final_video_id`; the first 10 MiB
cover all candidate IDs 0--7582. Downloading that prefix avoids fetching the
full multi-gigabyte catalog:

```bash
curl -L --fail --show-error \
  --range 0-10485759 \
  --output data/kuairand/KuaiRand-Pure/data/kuairand_video_categories.prefix.csv \
  https://zenodo.org/records/18159199/files/kuairand_video_categories.csv
```

Run the primary protocol with:

```bash
PYTHONPATH=src python -m matchope.pipeline.run_kuairand_matching_gate \
  --data-dir data/kuairand/KuaiRand-Pure/data \
  --category-metadata data/kuairand/KuaiRand-Pure/data/kuairand_video_categories.prefix.csv \
  --output-dir research/wsdm2027_matchope/evidence/kuairand_category_matching
```

`kuairand_category_matching_protocol.json` records the data hashes, action
mapping, pre-period model selection, estimand, nuisance protocol, and all point
and interval estimates. `kuairand_category_propensity_sensitivity.csv`
compares fold-excluded context frequencies with fixed category mass from the
candidate-pool metadata at TV=0.04. Alongside the cluster-$t$ intervals, the
protocol contains all
$2^{17}=131{,}072$ day-cluster sign flips. This is a small-cluster sensitivity
check under independent, roughly symmetric day clusters, not a replacement for
the primary interval. `kuairand_category_protocol_sensitivity.csv` changes one
design choice at a time around TV=0.04. Its nested context rows use 24-hour,
8-hour, and 4-hour bins. These runs diagnose sensitivity and were not used to
select the primary protocol. The 4-hour cells with no observed category are
finite-sample zero counts, not structural support holes under the random-pool
design.
