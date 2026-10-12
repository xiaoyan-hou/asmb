# ASMB-AD: PatchCore baseline

This repository contains a modular PatchCore baseline for MVTec AD. The first
target is the `bottle` category; ASMB and other selector variants are reserved
for later experiments.

## Setup

```powershell
conda env create -f environment.yml
conda activate asmb-ad
pip install -r requirements.txt
```

Put MVTec AD at `data/raw/mvtec/` with `bottle/train/good`,
`bottle/test`, and `bottle/ground_truth` directories. Run:

```powershell
python experiments/run_patchcore.py --config configs/patchcore_mvtec.yaml --category bottle
```

The command writes JSON metrics to `results/raw/` and sample visualizations to
`results/figures/`. Use `--batch-size` to reduce memory use when necessary.

## Design

`FeatureExtractor`, `MemorySelector`, `MemoryBank`, `Scorer`, `AnomalyMap`, and
`Evaluator` are independent components. PatchCore currently wires
`CoresetSelector` as the memory selector so future ASMB selectors can be
introduced without changing the pipeline.
