# Environment Setup

COSSAD requires GPU with at least 32 GB VRAM for training. We provide two setup options:
1. **Docker (recommended)**: Pre-configured containers for different GPU architectures
2. **Conda/Mamba**: Manual environment setup using YAML files

## Option 1: Docker (Recommended)

Docker provides the most reproducible environment with all dependencies pre-configured.

**Full documentation: [docker/README.md](../docker/README.md)**

### Quick Start

```bash
# Build image for your GPU
./docker/build.sh cu124    # For RTX 2060-4090, A100, H100
./docker/build.sh cu128    # For Blackwell (RTX 5090)

# Run inference (HOST_UID/HOST_GID ensure correct file ownership)
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -f docker/docker-compose.yml run --rm cossad-cu124 \
    python pipeline_reg_based.py --cossad_data_path=/data ...

# Interactive shell
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -f docker/docker-compose.yml run --rm cossad-cu124 bash
```

### Requirements

- Docker Engine 23.0+ with Compose V2
- NVIDIA Container Toolkit (for GPU access)

See [docker/README.md](../docker/README.md) for installation instructions.

### Running long jobs with tmux

Docker containers are tied to the terminal session that started them. If your SSH connection drops (common on remote machines like Lambda Labs), the container and its running process are killed. Use `tmux` on the **host** to keep the session alive.

Start tmux first, then launch Docker inside it:

```bash
tmux new -s cossad

# Now inside tmux — run Docker as usual
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -f docker/docker-compose.yml run --rm cossad-cu124 \
    python scripts/run_train_eval_pipeline.py full
```

Detach from tmux with `Ctrl+B`, then `D`. The Docker container keeps running. Re-attach later with `tmux attach -s cossad`.

Do **not** run tmux inside the container — it will die when the container stops.

## Option 2: Conda/Mamba

For Blackwell GPUs, use `cossad_blackwell.yaml`. For Ada-generation GPUs (e.g., A100), use 
`cossad_lambda_311.yaml`, which was verified on Lambda Labs.
Note that Blackwell setup requires manual steps listed at the end of `cossad_blackwell.yaml`.

The YAML files will set up `cossad_blackwell_311` or `cossad_lambda_311` environments, respectively. Once you set up 
the environment, activate it by running:
```bash
mamba activate cossad_blackwell_311
```
Then verify the setup is correct by running:
```bash
python ./scripts/verify_torch_environment.py
```
The expected output should include your Torch and CUDA versions, e.g.

```text
/home/iscander/.local/share/mamba/envs/cossad_blackwell_311/bin/python
/home/iscander/.local/lib/python3.11/site-packages
/home/iscander/cossad/scripts
/home/iscander/.local/share/mamba/envs/cossad_blackwell_311/lib/python311.zip
/home/iscander/.local/share/mamba/envs/cossad_blackwell_311/lib/python3.11
/home/iscander/.local/share/mamba/envs/cossad_blackwell_311/lib/python3.11/lib-dynload
/home/iscander/.local/share/mamba/envs/cossad_blackwell_311/lib/python3.11/site-packages
python: /home/iscander/.local/share/mamba/envs/cossad_blackwell_311/bin/python
torch: 2.8.0+cu128
torch cuda runtime: 12.8
cuda available: True
device 0: NVIDIA GeForce RTX 5090
device 0 capability: (12, 0)
PyG: 2.6.1
scatter_add works on CUDA: True
```

Contrastive encoder uses a modified RIConv++ backbone, whose source is located in the `external/riconv2` folder. It relies
on the `pointops` library of custom CUDA kernels for the point cloud operations, which was borrowed from the RISurConv
model. `pointops` will be compiled on-the-fly using your CUDA setup. We modified underlying pointops code
to make sure it works reliably on modern multi-GPU set-up. Unfortunately, pointops is fragile; if you encounter compilation
errors they are likely due to the mishaps in the environment setup (see above). 
