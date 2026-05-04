# COSSAD Reproduction Guide

This guide provides instructions to reproduce the results reported in the ICPR paper.
If you encounter any technical issues with the instructions below, please contact Alexander Tarvo (alextarvo at Gmail)
for assistance.

## Quick Start (Using Pre-trained Weights)

### 1. Setup Environment

We recommend use Docker for the minimal

```bash
# Clone repository
git clone --recurse-submodules https://github.com/alextarvo/cossad
cd cossad

# Build Docker image (cu124 recommended for most GPUs)
./docker/build.sh cu124
```

For Blackwell GPUs (RTX 5090), use `cu128` instead.

### 2. Download Dataset

Download and extract these files to your data directory (e.g., `/mnt/data/cossad`):
- https://pub-c3e1c2ecdbd44fc6bf5a8c091a3c3536.r2.dev/template.tar
- https://pub-c3e1c2ecdbd44fc6bf5a8c091a3c3536.r2.dev/train.tar

```bash
# Example: download to /mnt/data/cossad
cd /mnt/data/cossad
curl -O https://pub-c3e1c2ecdbd44fc6bf5a8c091a3c3536.r2.dev/template.tar
curl -O https://pub-c3e1c2ecdbd44fc6bf5a8c091a3c3536.r2.dev/train.tar
tar -xf template.tar
tar -xf train.tar
```

**Note:** If your data is in a different location, edit `docker/docker-compose.yml` and change the volume mount:
```yaml
volumes:
  - /your/data/path:/data    # Change this line
```

### 3. Download Pre-trained Weights

Pre-trained weights should be in `./weights/2025_12_11/` within the repository:
- `riconv2_split1.pth` through `riconv2_split12.pth`

These are automatically available inside the container at `/app/weights/2025_12_11/`.

### 4. Run Evaluation

```bash
# Set your user IDs for correct file ownership (add to ~/.bashrc to persist)
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# Run evaluation on Real3D-AD dataset
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -f docker/docker-compose.yml \
    run --rm cossad-cu124 ./scripts/evaluation_pretrained_v3_real3dad.sh

# Run evaluation on Anomaly-ShapeNet dataset
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -f docker/docker-compose.yml \
    run --rm cossad-cu124 ./scripts/evaluation_pretrained_v3_shapenet.sh
```

Alternatively, you may run from an interactive shell within Docker container itself:
```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -f docker/docker-compose.yml \
    run --rm cossad-cu124 bash

# Inside container:
./scripts/evaluation_pretrained_v3_real3dad.sh
./scripts/evaluation_pretrained_v3_shapenet.sh
```

Results are saved to `/data/predictions/<dataset>/2026_01_21/` (inside container) with CSV files 
containing O-ROCAUC and P-ROCAUC metrics. `all.csv` contains aggregated metrics for all objects.

Evaluation is described in the [Evaluation](evaluation.md) section of the documentation in more detail.

## Training from Scratch

### 1-2. Same as above (setup and dataset)

Additionally, download the training data:
```bash
cd /mnt/data/cossad
curl -O https://pub-c3e1c2ecdbd44fc6bf5a8c091a3c3536.r2.dev/supcon_train_r2_2025_11_04_real3dad_norot.tar
curl -O https://pub-c3e1c2ecdbd44fc6bf5a8c091a3c3536.r2.dev/supcon_train_r2_2025_11_04_shapenet_norot.tar
tar -xf supcon_train_r2_2025_11_04_real3dad_norot.tar
tar -xf supcon_train_r2_2025_11_04_shapenet_norot.tar
```

### 3. Train Models

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -f docker/docker-compose.yml \
    run --rm cossad-cu124 ./scripts/training_v3.bash
```

This trains 12 models (one per cross-validation fold). Training time: ~24-48 hours total.
The script generates a run ID in the format `Train_YYYYMMDD_HHMMSS` and prints it at the start and end of the run.

Model weights are saved to `/data/model_weights/<RUN_ID>/train_classes_N/best_riconv2.pth` inside the container, which maps to your data directory on the host (e.g., `/mnt/data/cossad/model_weights/<RUN_ID>/`).

Training is described in the [Training](training.md) section of the documentation in more detail.

### 4. Evaluate Your Trained Models

After training, evaluate using the pipeline runner with the run ID printed by the training script:

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -f docker/docker-compose.yml \
    run --rm cossad-cu124 ./scripts/evaluation_v3_real3dad.sh

HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -f docker/docker-compose.yml \
    run --rm cossad-cu124 ./scripts/evaluation_v3_shapenet.sh
```

**Note:** Weights are saved to `/data/model_weights/<RUN_ID>/` (the training output location),
not in `/app/weights/` (pre-trained weights location). Verify that `training_v3.bash` completed
successfully and weights exist before running evaluation.

## Understanding Results

- **12-fold cross-validation**: Each fold trains on 11/12 objects, tests on 1/12
- **Output**: CSV files in `/data/predictions/<dataset>/<date>/stats/all.csv`
- **Metrics**: O-ROCAUC (object-level), P-ROCAUC (point-level)
- **Multiple runs**: Scripts run 5 attempts per object for statistical significance

## Single Object Test

To quickly verify the setup works:

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -f docker/docker-compose.yml \
    run --rm cossad-cu124 python pipeline_reg_based.py \
        --cossad_data_path=/data \
        --object_class="airplane" \
        --encoder=riconv2 \
        --contrastive_encoder_path="./weights/2025_12_11/riconv2_split3.pth" \
        --patch_radius=2 \
        --filter_by_point_count \
        --output_results_path="/data/predictions/test"
```

## Requirements

- Docker Engine 23.0+ with Compose V2
- NVIDIA Container Toolkit
- GPU with ≥32 GB VRAM (for training) or ≥16 GB (for evaluation)
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
