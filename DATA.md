# Data sources and layout

Download the datasets from their publishers. This repository contains aggregate
experiment outputs, not the raw interaction logs. Run commands from the repository
root. The experiment script requires all four input locations explicitly.

## KuaiRec

Download `KuaiRec.zip` from [Zenodo record 18164998](https://zenodo.org/records/18164998).
The [official KuaiRec site](https://kuairec.com/) describes the dataset and its
paper. Extract the dense `small_matrix.csv` file to:

```text
data/kuairec/small_matrix.csv
```

The code reads user IDs, video IDs, and watch ratio from this
file. It constructs capped-watch-ratio and complete-watch outcomes. The sparse
`big_matrix.csv` and supplementary caption files are not required by this pipeline.
The saved eligibility audit reports the dense panel used by the paper; compare
that audit when checking a new input release.

## KuaiRand-Pure and category metadata

Download `KuaiRand-Pure.tar.gz` from
[Zenodo record 10439422](https://zenodo.org/records/10439422), linked from the
[official KuaiRand site](https://kuairand.com/). Extract it so that these two
logs are available:

```text
data/kuairand/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv
data/kuairand/KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv
```

The earlier standard log trains and selects the policy. The later randomized
log evaluates it. The intervention is an official first-level video category,
with the released within-category sampler retained; it is not exact-video OPE.

Obtain `kuairand_video_categories.csv` from the publisher's
[supplementary record 18159199](https://zenodo.org/records/18159199). The paper
uses the first 10 MiB: the file is ordered by `final_video_id`, and this prefix
covers candidate video IDs 0–7582. The existing protocol checks candidate coverage.

```bash
mkdir -p data/kuairand/KuaiRand-Pure/data
curl -L --fail --show-error \
  --range 0-10485759 \
  --output data/kuairand/KuaiRand-Pure/data/kuairand_video_categories.prefix.csv \
  https://zenodo.org/records/18159199/files/kuairand_video_categories.csv
```

The expected prefix size is 10,485,760 bytes. The publisher's checksum for the
complete category file does not apply to this partial download. Existing input
checksums and relative paths are retained in
[`kuairand_category_matching_protocol.json`](research/wsdm2027_matchope/evidence/kuairand_category_matching/kuairand_category_matching_protocol.json)
under `data_audit`. See the
[saved protocol notes](research/wsdm2027_matchope/evidence/kuairand_category_matching/README.md)
for category pooling, overlap, and sensitivity definitions.

## Open Bandit Dataset (OBD)

Follow the [official dataset instructions](https://github.com/st-tech/zr-obp/blob/master/obd/README.md)
and obtain the **full release** from the linked
[ZOZO Research data page](https://research.zozo.com/data.html).
The small CSVs bundled with the Open Bandit Pipeline repository have only
10,000 records per campaign/policy pair and do not reproduce these results.
Extract the full random-policy men's campaign to:

```text
data/open_bandit_dataset_full/open_bandit_dataset/random/men/men.csv
```

The analysis selects position one, giving the saved design of 150,936 rows
over seven dates. It uses the 34 released user–item affinity features.
No other OBD campaign is needed.

## Dataset licenses and citation

The KuaiRec and KuaiRand records above specify CC BY 4.0. OBD is also released
under CC BY 4.0; consult the publisher's terms and citation instructions before
reuse. The repository's MIT license applies to project code and documentation,
not these external datasets. Cite the original KuaiRec, KuaiRand, and Open Bandit
Dataset papers as appropriate. Original author lists and citation entries are
available at the linked publisher pages.
