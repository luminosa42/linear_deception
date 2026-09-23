# Detecting Strategic Deception: Small-Scale Replication + Causal-Use Extension

This is a research project for the Bluedot technical AI safety project sprint. 
The main aim is to replicate the core methodology of Goldowsky-Dill et al., 2025 (Apollo Research), "Detecting Strategic Deception Using Linear Probes," at small scale, 
and extending it with a causal-use test the original paper leaves open.

## Motivation

The original paper trains linear probes on a model's internal activations to detect whether it is being deceptive, and reports strong separation (AUROC 0.96–0.999) on realistic held-out scenarios. 
This is a correlational result: it shows deception-related information is linearly decodable from activations, not that the model's behavior is causally produced by that direction. This project:

- Replicates the core probing methodology at a laptop-friendly scale (Qwen2.5-1.5B-Instruct, CPU).
- Extends it with an activation patching experiment to test whether the probe direction is causally load-bearing, or merely a decodable correlate — the central open question the original paper does not test.

## Project structure

Each file in `src/` is a standalone Jupyter notebook covering one phase of the
pipeline. They run in order, and each one's outputs are cached to disk so a later
notebook can be run on its own without repeating the expensive steps.

| Notebook | Phase | Writes |
| --- | --- | --- |
| `smoke_test.ipynb` | Environment check: model loads, generation, cache shape, chat template | — |
| `build_data.ipynb` | Phase 1 — generate honest/deceptive response pairs and split them | `data/probe_{train,eval}.json` |
| `probe_pipeline_main.ipynb` | Phase 2 — extract activations, train one linear probe per layer | `data/features_*.npz`, `data/baseline_training_summary.json` |
| `plotting_paper.ipynb` | Figures for comparison with the paper | `plots/*.png` |
| `intervention.ipynb` | Phase 3 — activation patching (ablation / injection + random control) | `data/intervention_results_*.json`, `plots/*.png` |

`src/common.py` holds the helpers the notebooks share: path setup, model loading,
activation extraction, cached-feature loading and probe fitting.

The slow, output-overwriting cells (dataset generation, feature extraction, the
intervention sweeps) are left commented out, since their results are already
cached in `data/`. Uncomment them to regenerate.

### Running

Select the `interp` conda environment as the notebook kernel. The notebooks
locate the repo root themselves and `chdir` there, so they work regardless of
where the kernel starts.
