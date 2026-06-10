# Training & Validation Pipeline

`scripts/run_train_eval_pipeline.py` orchestrates the COSSAD train/eval/compare pipeline. It calls `contrastive_learner_v3.py`, `pipeline_reg_based.py`, and `scripts/compare_models.py` as subprocesses.

Run from the repo root:

```bash
python scripts/run_train_eval_pipeline.py <command> [options]
```

---

## Overview

The pipeline automates the full COSSAD workflow: cross-validation training, inference, and statistical comparison against a baseline. Each stage operates on one cross-validation split independently.

```
train@1 ──┐
train@2 ──┤
   ...     ├─→ eval@1 ──┐
train@N ──┘    eval@2 ──┤
                  ...    ├─→ compare  (after all evals)
               eval@N ──┘
```

- **train@N** — trains a contrastive encoder for split N via `contrastive_learner_v3.py` (Hydra). Skips if weights already exist.
- **eval@N** — runs inference on held-out test objects for split N via `pipeline_reg_based.py` (argparse). Always runs when invoked.
- **compare** — statistical comparison (Mann-Whitney U) of eval results against baseline via `scripts/compare_models.py`. Runs after all eval stages complete.

Stages can be run together (preset tiers: smoke, fast, full) or independently (train alone, then eval+compare separately). This supports the common workflow of training once and re-evaluating multiple times with different settings.

---

## Run IDs

Every pipeline run is identified by two run IDs — short string tokens that determine where outputs are written and serve as the primary handle for referencing past runs.

The **weights run ID** identifies a training run. It determines where trained model weights are stored. Format: `Train_YYYYMMDD_HHMMSS` (e.g. `Train_20260320_021222`).

The **eval run ID** identifies an evaluation run. It determines where inference predictions and statistics are stored. Format: `Eval_YYYYMMDD_HHMMSS` (e.g. `Eval_20260320_032141`).

Training and evaluation have separate run IDs because they have independent lifecycles: you train once, then evaluate multiple times with different settings. Each evaluation gets its own run ID and output directory.

### Automatic generation

By default, the runner generates both run IDs from the current timestamp. In a full run (preset tier), both share the same date-time suffix:

```
Weights run ID:  Train_20260320_021222
Eval run ID:     Eval_20260320_021222
```

Both are printed at the start of every run. Save them if you need to reference the run later.

Automatic generation is primarily intended for testing: run the pipeline, check that the model works, and compare accuracy to the baseline.

### Overriding run IDs

For experimentation, either run ID can be overridden via CLI flags. This is useful when re-evaluating an existing model with different parameters:

```bash
# Override both (e.g. to reproduce an exact run):
python scripts/run_train_eval_pipeline.py full \
    --weights-run-id Train_20260320_021222 \
    --eval-run-id Eval_20260320_021222

# Reuse trained weights with a fresh eval (most common):
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id Train_20260320_021222
```

When `--eval-run-id` is not specified, a fresh ID is generated automatically, so each re-evaluation gets its own output directory.

### Re-evaluating without retraining

When training has already completed, you can run evaluation alone by providing the weights run ID:

```bash
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id Train_20260320_021222 \
    --splits 2,3 --attempts 1 --eval-dataset real3dad
```

The runner checks for existing weight files before each split. Splits whose `best_riconv2.pth` is present are available for eval; splits missing weights cause eval to fail with a clear error rather than silently skipping.

### Output layout

Run IDs map directly to directory paths:
- Model weights: `<weights-root>/<weights-run-id>/train_classes_N/best_riconv2.pth`
- Predictions and stats: `<predictions-root>/<eval-run-id>/stats/all.csv`
- Comparison results: `<predictions-root>/<eval-run-id>/comparison_results.csv`
- Baseline (reference): `<predictions-root>/baseline/stats/all.csv`

---

## Training skip logic

Before training each split, the runner checks if `<weights-root>/<weights-run-id>/train_classes_N/best_riconv2.pth` exists. If it does, that split is skipped. This is based on the sentinel file pattern: `contrastive_learner_v3.py` writes `best_riconv2.tmp` during training and renames it to `.pth` only on clean completion. A failed or interrupted training run leaves only `.tmp`, preventing eval from running against incomplete weights.

If only some splits trained successfully (e.g. training was interrupted), re-running with the same `--weights-run-id` will train only the missing splits and skip the rest. There is no need to clean up or restart from scratch.

To force retraining a split, delete its weights directory:

```bash
rm -rf /data/model_weights/Train_20260320_151240/train_classes_3/
```

Eval has no skip logic — it always runs when invoked.

## Error handling

- If a training split fails, the pipeline stops and does not proceed to eval.
- If an eval split fails, the pipeline stops and does not proceed to compare.
- If compare exits non-zero (regression detected), this is reported but does not count as a pipeline failure — the user decides what to do.
- If training exits 0 but the sentinel `.pth` file is not found, the runner treats it as a failure.
- If eval is invoked but the required weights file does not exist, it fails with a clear error message rather than silently skipping.

## Log file

Each execution writes a log file to `<predictions-root>/<eval-run-id>/run_log.txt`. The log contains:

- Run IDs and all resolved parameters
- Full command line of every subprocess invocation (copy-pasteable)
- Exit codes and sentinel file check results (trained vs skipped per split)
- Summary of all stage outcomes

The log mirrors the console output exactly. In `--dry-run` mode, the log contains the same command lines but marked as not executed.

---

## Design rationale

### Why a single Python script

The pipeline runner handles parameterized invocation and stage-skipping in plain Python without external tooling. WandB handles experiment tracking. This keeps the pipeline self-contained with no dependencies beyond the standard library.

### Why subprocess calls, not imports

Training and eval are invoked via `subprocess.run()`, not by importing their modules. This preserves Hydra's config system for training and avoids module-level side effects from either script.

### Why no config files

Everything the runner needs is supplied via CLI arguments. Preset tiers (smoke/fast/full) are hardcoded in the script. No external config files to keep in sync.

### Why compare is non-fatal

The compare stage exits non-zero when the test mean is below baseline (regression). The runner reports this clearly but does not treat it as a pipeline failure. A regression may be expected (e.g. testing a smaller model), and the user should decide whether to act on it.

### Run ID format and quoting

Run IDs follow the format `Train_YYYYMMDD_HHMMSS`. The underscore is significant — without single-quote wrapping, Hydra's YAML parser interprets the numeric portions as integers and silently drops the underscore. The runner wraps run IDs in single quotes when passing them as Hydra overrides (`experiment.run_id='Train_...'`).

---

## Commands

There are two kinds of commands: **preset tiers** and **individual stages**.

### Preset tiers

Preset tiers run the full pipeline (train + eval + compare) with predefined parameters:

```bash
python scripts/run_train_eval_pipeline.py smoke   # 2 splits, 10 epochs, 1 attempt
python scripts/run_train_eval_pipeline.py fast    # 3 splits, 50 epochs, 3 attempts
python scripts/run_train_eval_pipeline.py full    # 12 splits, 50 epochs, 5 attempts
```

| Tier  | Splits    | Epochs | Attempts | Train datasets         | Eval dataset | Purpose                  |
|-------|-----------|--------|----------|------------------------|--------------|--------------------------|
| smoke | 2, 3      | 10     | 1        | real3dad, shapenet     | real3dad     | Smoke testing            |
| fast  | 2, 3, 4   | 50     | 3        | real3dad, shapenet     | real3dad     | Regular testing          |
| full  | 1 through 12 | 100    | 5        | real3dad, shapenet     | real3dad     | Full experimentation run |

### Individual stages

Run stages independently for more control:

```bash
# Train only
python scripts/run_train_eval_pipeline.py train --splits 2,3,4 --epochs 50

# Eval + compare using existing weights
python scripts/run_train_eval_pipeline.py eval --weights-run-id Train_20260320_151240 --splits 2,3

# Compare only
python scripts/run_train_eval_pipeline.py compare --eval-run-id Eval_20260320_151240
```

- `train` runs training only.
- `eval` runs evaluation + comparison. Requires `--weights-run-id` to point at existing trained weights.
- `compare` runs comparison only. Requires `--eval-run-id` to point at existing eval results.

## Options

### Run ID overrides

```
--weights-run-id ID    Where trained weights are stored (default: Train_YYYYMMDD_HHMMSS)
--eval-run-id ID       Where eval outputs are stored (default: Eval_YYYYMMDD_HHMMSS)
```

### Stage parameters

```
--splits SPLITS            Comma-separated list or range: "2,3,4" or "1-12"
--epochs N                 Training epochs (overrides preset default)
--attempts N               Eval inference attempts per class (overrides preset default)
--train-datasets DATASETS  Comma-separated: "real3dad,shapenet"
--eval-dataset DATASET     Single dataset name: "real3dad" or "shapenet"
```

### Path overrides

Default paths assume the Docker container volume at `/data`. Override for local or remote runs:

```
--data-path PATH           Base data directory (default: /data)
--weights-root PATH        Where model weights are stored (default: /data/model_weights)
--predictions-root PATH    Where eval outputs are stored (default: /data/predictions)
```

### Passthrough arguments for ablation studies

```
--train-extra ARGS   Extra arguments passed verbatim to contrastive_learner_v3.py (Hydra overrides)
--eval-extra ARGS    Extra arguments passed verbatim to pipeline_reg_based.py (argparse flags)
```

These flags allow passing arbitrary arguments to the underlying training or evaluation scripts without modifying the pipeline runner. The value is a quoted string that gets split on whitespace and appended to the subprocess command.

This is designed for **ablation studies**: run the same trained model through multiple evaluation configurations to measure the effect of individual parameters. For example, testing sensitivity to registration perturbation at different levels.

```bash
# Evaluate with 5° rotation perturbation
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id Train_20260320_151240 \
    --eval-run-id Eval_rot5 \
    --eval-extra '--perturb_rotation_deg 5 --perturb_translation_percent 0'

# Evaluate with 10% translation perturbation
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id Train_20260320_151240 \
    --eval-run-id Eval_trans10 \
    --eval-extra '--perturb_rotation_deg 0 --perturb_translation_percent 10'

# Pass extra Hydra overrides to training
python scripts/run_train_eval_pipeline.py train \
    --train-extra 'training.lr=0.001 model.dropout=0.2'
```

Use `--eval-run-id` to give each ablation a distinct output directory so results don't overwrite each other.

### Dry run

```
--dry-run    Print all commands without executing them
```

The dry-run output is copy-pasteable. Each printed command can be run manually in the terminal. A log file is still written.

## Common workflows

### Full cross-validation from scratch

```bash
python scripts/run_train_eval_pipeline.py full
```

Trains all 12 splits, evaluates each, then compares against baseline.

### Quick sanity check

```bash
python scripts/run_train_eval_pipeline.py smoke
```

Trains 2 splits with 10 epochs, runs eval with 1 attempt. Good for verifying the pipeline works end-to-end after code changes.

### Re-evaluate with different settings

Train once, then re-evaluate multiple times:

```bash
# Train
python scripts/run_train_eval_pipeline.py train --splits 1-12 --epochs 50

# Note the weights run ID from the output, e.g. Train_20260320_151240

# Re-evaluate with 1 attempt (fast)
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id Train_20260320_151240 \
    --splits 1-12 --attempts 1

# Re-evaluate with 5 attempts (thorough)
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id Train_20260320_151240 \
    --splits 1-12 --attempts 5
```

### Ablation study over a parameter

Train once, then evaluate multiple times varying a single parameter. Use `--eval-run-id` to store each ablation's results separately, and `--eval-extra` to pass the varying parameter:

```bash
# Train
python scripts/run_train_eval_pipeline.py train --splits 1-12 --epochs 50
# → Weights run ID: Train_20260401_090000

# Baseline eval (no perturbation)
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id Train_20260401_090000 \
    --eval-run-id Eval_rot0 \
    --splits 1-12 --attempts 5

# Ablation: 5° rotation perturbation
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id Train_20260401_090000 \
    --eval-run-id Eval_rot5 \
    --splits 1-12 --attempts 5 \
    --eval-extra '--perturb_rotation_deg 5'

# Ablation: 10° rotation perturbation
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id Train_20260401_090000 \
    --eval-run-id Eval_rot10 \
    --splits 1-12 --attempts 5 \
    --eval-extra '--perturb_rotation_deg 10'
```

Each eval run writes to its own directory under `<predictions-root>/Eval_rot*/`, making it easy to compare results across ablation levels.

### Resume interrupted training

If training was interrupted, just re-run the same command. Splits that already produced a `best_riconv2.pth` file are skipped automatically. Only missing splits will train.

```bash
# Original run was interrupted after split 5
python scripts/run_train_eval_pipeline.py full --weights-run-id Train_20260320_151240
# Splits 1-5 are skipped, 6-12 train normally
```

### Verify commands before running

```bash
python scripts/run_train_eval_pipeline.py full --dry-run
```

Review the exact commands that will be executed. Copy-paste any individual command to run it manually.

### Run on local machine (outside Docker)

```bash
python scripts/run_train_eval_pipeline.py smoke \
    --data-path /mnt/data/cossad \
    --weights-root /mnt/data/cossad/model_weights \
    --predictions-root /mnt/data/cossad/predictions
```

## Configurable constants

All flags passed to the training and eval scripts are defined as constants at the top of `scripts/run_train_eval_pipeline.py`. Update these when the Hydra config or eval script arguments change:

| Constant | Value | Used by |
|----------|-------|---------|
| `TRAIN_CONFIG_PATH` | `configs` | Hydra `--config-path` |
| `TRAIN_CONFIG_NAME` | `config_riconv_msloss_v3` | Hydra `--config-name` |
| `TRAIN_MODEL_NAME` | `riconv2` | Sentinel filename: `best_{name}.pth` |
| `TRAIN_RUN_NAME_PREFIX` | `riconv` | WandB run name: `{prefix}_{split}` |
| `TRAIN_PATCHES_SUBDIR` | `supcon_train_r2_2025_11_04` | Training data subdirectory |
| `TRAIN_EXPERIMENT_TAGS` | `[v3,1_tensor_per_batch,simple_miner]` | WandB tags |
| `EVAL_ENCODER` | `riconv2` | `--encoder` flag for eval |
| `EVAL_PATCH_RADIUS` | `2` | `--patch_radius` flag for eval |

Dataset-specific eval flags are defined in `DATASET_EVAL_FLAGS`:

```python
DATASET_EVAL_FLAGS = {
    "real3dad": ["--filter_by_point_count"],
    "shapenet": [],
}
```

## Relationship to other files

- **`contrastive_learner_v3.py`** — called by the train stage as a subprocess via Hydra. Not modified by the runner.
- **`pipeline_reg_based.py`** — called by the eval stage as a subprocess via argparse. Not modified by the runner.
- **`scripts/compare_models.py`** — called by the compare stage. Performs Mann-Whitney U statistical comparison.
- **WandB** — experiment tracking is handled inside `contrastive_learner_v3.py`. The runner does not interact with WandB.
