# Evidence weighting in direct-answer models

Direct-answer models can read individual reports correctly while giving a probability that responds too little to the amount of evidence. This repository studies that behavior with synthetic reports whose correct posterior is known.

With 48 of 64 independent reports favoring an outcome, each 55% reliable, the specified posterior is 0.998. Jev's Noul readout averaged 0.762 (95% bootstrap interval 0.732 to 0.788). This result concerns the Noul probability readout; other Jev output formats can behave differently. See the [results](experiments/evidence_count/results/summary.md) for the full analysis.

![Specified posterior and Jev's Noul probability as the number of independent reports increases](figures/evidence.png)

## Contents

| Directory | Contents |
| --- | --- |
| [`experiments/evidence_count/`](experiments/evidence_count/) | Independent reports, a decisive-report control, prompts, analysis, and results |
| [`experiments/pair_matching/`](experiments/pair_matching/) | Matched-pair detection in longer rosters |
| [`experiments/product_reviews/`](experiments/product_reviews/) | A check using held-out Amazon review ratings |
| [`experiments/open_models/`](experiments/open_models/) | Open-model probes, interventions, and synthetic-data fine-tuning |
| [`api/`](api/) | Hosted-model clients and usage logging |

Each experiment includes code and released aggregate results under `results/`. `plot_evidence.py` draws the main figure from the released metrics. The open-model interventions are inconclusive because their positive control failed; the fine-tuning comparison is exploratory.

## Quick start

Use Python 3.12 or newer. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s experiments/evidence_count -p 'test_generate.py'
python -m pytest -q experiments/pair_matching/test_generate.py
python -m unittest discover -s experiments/product_reviews -p 'test_sample.py'
python experiments/evidence_count/generate.py --output experiments/evidence_count/cases.jsonl
python experiments/pair_matching/generate.py
python plot_evidence.py
```

`plot_evidence.py` writes `figures/evidence.pdf` and `figures/evidence.png` from the released metrics. These commands make local cases and plots without model access. [Example prompts](experiments/evidence_count/results/examples.md) are also included.

The reference log-odds are `(2k - n) log(r / (1-r))` for `n` reports, `k` positive reports, and reliability `r`. The calculation assumes an equal prior, conditionally independent reports, and identical symmetric reliability.

## Model runs and data

- **Independent reports and pair matching:** See [`evidence_count/run.py`](experiments/evidence_count/run.py) and [`pair_matching/run.py`](experiments/pair_matching/run.py). Each accepts `--help`. Set `TYPESAFE_API_KEY`, `OPENROUTER_API_KEY`, or `ANTHROPIC_API_KEY` for the selected model. Responses go to the experiment's `results/raw/` directory.
- **Product reviews:** Run `experiments/product_reviews/sample.py`, then `extend_sample.py`, to sample the `Health_and_Personal_Care` category of [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). Held-out ratings provide an empirical target, unlike the exact synthetic posterior. The download follows the dataset's `main` revision, which may change.
- **Open models:** `experiments/open_models/modal_app.py` runs the GPU experiments on Modal and fetches [Kev](https://github.com/jaredpalmer/kev) at commit `d32a973cc2375f1c76e898cbf6e759c4499042de`. Fine-tuning labels come from the synthetic probability rule, not Jev outputs.

Historical hosted-model responses, downloaded reviews, activations, and checkpoints are not included. The aggregate results can be inspected and plotted, but recalculating historical estimates requires the omitted raw responses. Fresh runs produce new measurements.

## License

MIT. See [LICENSE](LICENSE).
