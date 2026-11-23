# MatchOPE — WSDM 2027

Code and saved experimental results for **Higher Policy Value Can Mask Worse
User–Item Matching: Controlling Exposure in Off-Policy Evaluation** (WSDM 2027).

**Xiaoyu Zhou, Haifeng Li, and Mo Hai** · Central University of Finance and Economics

WSDM 2027, Main Track · [Paper DOI: 10.1145/3862322.3863623](https://doi.org/10.1145/3862322.3863623)

MatchOPE measures the value of user–item matching by comparing a target policy
with its user-averaged reference within shared contexts. The comparison keeps
the policy's exposure distribution fixed. A policy can improve total reward
by exposing better items while making worse user–item assignments; the code
separates these effects and implements support analysis and statistical inference.

This repository contains the paper's implementation, tests, aggregate results,
and experiment commands. The [paper bundle](paper/MatchOPE_WSDM2027.zip) includes
the main and supplementary PDFs, LaTeX sources, and required typesetting files.
Raw datasets are not included.

## Install and run a small example

Use Python 3.10 and run the commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python examples/two_users.py
PYTHONPATH=src python -m pytest -q
```

The example needs no downloaded data. Two users prefer different items; the
target assigns each user their preferred item, while the reference gives both
users the same item probabilities. It prints:

```text
Target value: 1.0
Reference value: 0.5
Matching value: 0.5
```

## Reproduce the experiments

1. Obtain the inputs listed in [DATA.md](DATA.md).
2. Run the full pipeline or the individual commands in
   [REPRODUCE.md](REPRODUCE.md).
3. Use [RESULTS.md](RESULTS.md) to map paper figures and tables to saved results
   and their generating modules.

The full pipeline has 11 experiment and plotting stages. To inspect its commands
without loading data or writing results:

```bash
bash scripts/run_matchope_paper_pipeline.sh \
  --kuairec-csv data/kuairec/small_matrix.csv \
  --kuairand-data-dir data/kuairand/KuaiRand-Pure/data \
  --kuairand-category-metadata data/kuairand/KuaiRand-Pure/data/kuairand_video_categories.prefix.csv \
  --obd-csv data/open_bandit_dataset_full/open_bandit_dataset/random/men/men.csv \
  --dry-run
```

Remove `--dry-run` to run it. The commands write into the saved evidence
directory and overwrite files with the same names. Run in a separate copy if
you want to compare new outputs with the published results.

| Path | Contents |
| --- | --- |
| `src/matchope/eval/matching_value.py` | Matching value, reference policies, support and inference utilities |
| `src/matchope/pipeline/` | Synthetic, KuaiRec, KuaiRand, OBD, and plotting modules |
| `tests/` | Existing tests for estimators and experiment components |
| `examples/two_users.py` | Data-free example |
| `research/wsdm2027_matchope/evidence/` | Saved aggregate CSVs, protocol JSONs, and figures |
| `scripts/run_matchope_paper_pipeline.sh` | Full experiment and plotting entry point |

The saved results include null, nonsignificant, and sensitivity findings.
Two compact report summaries are retained but are not regenerated automatically;
[RESULTS.md](RESULTS.md#saved-report-summaries) identifies their detailed sources.

## Citation and license

Please cite the paper when using this code. [CITATION.cff](CITATION.cff) also
provides the paper as the preferred citation for GitHub's citation menu.

```bibtex
@inproceedings{zhou2027matchope,
  author = {Zhou, Xiaoyu and Li, Haifeng and Hai, Mo},
  title = {Higher Policy Value Can Mask Worse User--Item Matching:
           Controlling Exposure in Off-Policy Evaluation},
  booktitle = {Proceedings of the Twentieth ACM International Conference
               on Web Search and Data Mining},
  series = {WSDM '27},
  year = {2027},
  doi = {10.1145/3862322.3863623},
  url = {https://doi.org/10.1145/3862322.3863623}
}
```

Project code and documentation use the [MIT license](LICENSE). The paper bundle
retains the paper and template licenses stated in its files. External datasets
retain their own licenses and should also be cited; see [DATA.md](DATA.md).
