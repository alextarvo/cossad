# COSSAD Reproduction Guide

This guide provides instructions to reproduce the results reported in the ICPR'26 paper "The Good, the Bad, and the Template: Contrastive Anomaly Detection in 3D".
If you encounter any technical issues with the instructions below, please contact Alexander Tarvo (alextarvo at Gmail) for assistance.

## Quick Start (Using Pre-trained Weights)

### 1. Setup Environment

You need a GPU 32 Gb VRAM to train the COSSAD with the default settings. We trained COSSAD on NVidia RTX5090 GPU, so we recommend Blackwell architecture
for ultimate reproducibility. Inference is a CPU-heavy task due to the ICP registration algorithm, thus we recommend an Intel i7 or Core 7 CPU with 8 performance cores.

We provided Docker environment for reproducibility. To build a Docker environment:

```bash
# Clone repository
git clone --recurse-submodules https://github.com/alextarvo/cossad
cd cossad

# Build Docker image (cu124 recommended for most GPUs)
./docker/build.sh cu124
```

For Blackwell GPUs (RTX 5090), use

```
./docker/build.sh cu128
```

 instead.

### 2. Download Dataset

Download and extract these files to your data directory (e.g., `/mnt/data/cossad/data`):
- https://pub-c3e1c2ecdbd44fc6bf5a8c091a3c3536.r2.dev/template.tar
- https://pub-c3e1c2ecdbd44fc6bf5a8c091a3c3536.r2.dev/train.tar

```bash
cd /mnt/data/cossad/data
curl -O https://pub-c3e1c2ecdbd44fc6bf5a8c091a3c3536.r2.dev/template.tar
curl -O https://pub-c3e1c2ecdbd44fc6bf5a8c091a3c3536.r2.dev/train.tar
tar -xf template.tar
tar -xf train.tar
```

If your data is in a different location, update the volume mounts in `docker/docker-compose.yml`:
```yaml
volumes:
  - /your/data/path:/data          # Dataset (template/ and train/)
  - /your/weights/path:/weights    # Pre-trained model weights
```

For example, if your data and weights are in the `/mnt/data/cossad/data/`  and `/mnt/data/cossad/model_weights/`, then
your volumes record should be:

```yaml
volumes:
  - /mnt/data/cossad/data:/data          # Dataset (template/ and train/)
  - /mnt/data/cossad/model_weights:/weights    # Pre-trained model weights
```



### 3. Copy Pre-trained Weights

Copy the pre-trained weights from the ./weights in the repository folder to your destination weights folder, `/mnt/data/cossad/model_weights/`.
The directory must follow this structure:

```
/mnt/data/cossad/data/model_weights/
  baseline/
    train_classes_1/best_riconv2.pth
    train_classes_2/best_riconv2.pth
    ...
    train_classes_12/best_riconv2.pth
```

This directory is mounted into the container at `/weights/` mount point. The pipeline runner references
the pre-trained weights as `--weights-run-id baseline`.

### 4. Start Docker Container

All commands below run inside the Docker container. From the repository root (`cossad/`),
start an interactive shell:

```
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -f docker/docker-compose.yml \
    run --rm cossad-cu128 bash
```

**Tip:** Add `export HOST_UID=$(id -u)` and `export HOST_GID=$(id -g)` to your `~/.bashrc`
to avoid typing them each time.

### 5. Run Evaluation

**Dry-run (verify commands without executing):**

An optional step. Here the script will show what the commands will be executed.

```bash
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id baseline --eval-dataset real3dad --attempts 5 --splits 1-12 --dry-run
```
**Evaluate on Real3D-AD dataset:**

```bash
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id baseline --eval-dataset real3dad --attempts 5 --splits 1-12
```

**Evaluate on Anomaly-ShapeNet dataset:**

```bash
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id baseline --eval-dataset shapenet --attempts 5 --splits 1-12
```

The `--attempts 5` flag runs inference 5 times per object class and aggregates the results. 
The inference algorithm is inherently stochastic.
Running multiple attempts provides mean and standard deviation
of the ROCAUC metrics, which is necessary to obtain the averaged accuracy values and
assess statistical significance of the results.

Results are saved to `/data/predictions/<EVAL_RUN_ID>/stats/all.csv` inside the container,
where `<EVAL_RUN_ID>` is printed at the start of the run (format: `Eval_YYYYMMDD_HHMMSS`).
The CSV contains O-ROCAUC and P-ROCAUC metrics aggregated across all object classes.

Evaluation is described in the [Evaluation](evaluation.md) section of the documentation in more detail.

## Training from Scratch

### 1-3. Same as above (setup, dataset, and start Docker container)

Additionally, before starting the container, download the training data:
```bash
cd /mnt/data/cossad/data
curl -O https://pub-c3e1c2ecdbd44fc6bf5a8c091a3c3536.r2.dev/supcon_train_r2_2025_11_04_real3dad_norot.tar
curl -O https://pub-c3e1c2ecdbd44fc6bf5a8c091a3c3536.r2.dev/supcon_train_r2_2025_11_04_shapenet_norot.tar
tar -xf supcon_train_r2_2025_11_04_real3dad_norot.tar
tar -xf supcon_train_r2_2025_11_04_shapenet_norot.tar
```

### 4. Train and Evaluate

The `full` preset trains 12 models (one per cross-validation fold), evaluates on Real3D-AD,
and compares against baseline:

```bash
python scripts/run_train_eval_pipeline.py full
```

Training time: ~24-48 hours total depending on the GPU.

The runner generates and prints two run IDs at the start:
- `Train_YYYYMMDD_HHMMSS` — where trained weights are saved
- `Eval_YYYYMMDD_HHMMSS` — where evaluation results are saved

Model weights are saved to `/weights/<TRAIN_RUN_ID>/train_classes_N/best_riconv2.pth` inside the
container, which maps to `/mnt/data/cossad/model_weights/<TRAIN_RUN_ID>/` on the host.

For a quick sanity check (2 splits, 10 epochs), use `smoke` instead of `full`:

```bash
python scripts/run_train_eval_pipeline.py smoke
```

Training is described in the [Training](training.md) section of the documentation in more detail.

### 5. Evaluate Your Trained Models Separately

To re-evaluate existing weights on a different dataset or with different parameters,
use the `eval` command with the training run ID:

```bash
# Evaluate on Real3D-AD
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id Train_YYYYMMDD_HHMMSS --eval-dataset real3dad --attempts 5 --splits 1-12

# Evaluate on Anomaly-ShapeNet
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id Train_YYYYMMDD_HHMMSS --eval-dataset shapenet --attempts 5 --splits 1-12
```

Replace `Train_YYYYMMDD_HHMMSS` with the actual run ID printed by the training step.

## Understanding Results

- **12-fold cross-validation**: Each fold trains on 11/12 objects, tests on 1/12
- **Output**: CSV files in `/data/predictions/<EVAL_RUN_ID>/stats/all.csv`
- **Metrics**: O-ROCAUC (object-level), P-ROCAUC (point-level)
- **Multiple runs**: 5 inference attempts per object for statistical significance

## Single Object Test

To quickly verify the setup works (inside the container):

```bash
python pipeline_reg_based.py \
    --cossad_data_path=/data \
    --object_class="airplane" \
    --encoder=riconv2 \
    --contrastive_encoder_path="/weights/baseline/train_classes_3/best_riconv2.pth" \
    --patch_radius=2 \
    --filter_by_point_count \
    --output_results_path="/data/predictions/test"
```

## Pipeline Runner Reference

The pipeline runner (`scripts/run_train_eval_pipeline.py`) supports the following commands:

| Command | Description |
|---------|-------------|
| `smoke` | Quick sanity check: 2 splits, 10 epochs, 1 attempt |
| `fast` | Moderate run: 4 splits, 50 epochs, 3 attempts |
| `full` | Full 12-fold cross-validation: 12 splits, 120 epochs, 5 attempts |
| `train` | Training only (no evaluation) |
| `eval` | Evaluation + comparison only (requires `--weights-run-id`) |
| `compare` | Comparison against baseline only (requires `--eval-run-id`) |

Key flags: `--splits`, `--epochs`, `--attempts`, `--eval-dataset`, `--weights-run-id`,
`--eval-run-id`, `--data-path`, `--weights-root`, `--dry-run`.

See `python scripts/run_train_eval_pipeline.py --help` for the full list.

## Requirements

- Docker Engine 23.0+ with Compose V2
- NVIDIA Container Toolkit
- GPU with >=32 GB VRAM (for training) or >=16 GB (for evaluation)
- ~100 GB disk space

See [docker/README.md](../docker/README.md) for Docker installation instructions.

## Troubleshooting

**First run is slow:** On first container start, the `pointops` CUDA extension compiles (~1-2 min).
This is cached for subsequent runs.

**Permission errors:** Ensure `HOST_UID` and `HOST_GID` are set correctly. Add to `~/.bashrc`:
```bash
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)
```

**GPU not detected:** Verify NVIDIA Container Toolkit is installed:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

For detailed information, see:
- [Training](training.md) - Training details
- [Evaluation](evaluation.md) - Evaluation details
- [Environment setup](environment.md) - Environment setup options
- [Docker README](../docker/README.md) - Full Docker documentation
