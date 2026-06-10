# COSSAD - Unified Dockerfile
#
# REQUIRES: Docker 23.0+ (BuildKit). See docker/README.md
#
# Variants (configured via build args):
#   cu124 - CUDA 12.4 + PyTorch 2.5 (Lambda Labs A100/H100, RTX 2060-4090)
#   cu128 - CUDA 12.8 + PyTorch 2.8 (Blackwell RTX 5090)
#
# Build via docker compose (recommended):
#   docker compose -f docker/docker-compose.yml build cossad-cu124
#   docker compose -f docker/docker-compose.yml build cossad-cu128

# ---------------------------------------------------------------------------
# Build arguments (REQUIRED - no defaults, must be passed via docker-compose.yml)
# ---------------------------------------------------------------------------
# CUDA_VERSION must be before FROM to select base image
ARG CUDA_VERSION

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# Remaining args declared here (only place they're needed)
ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION
ARG CUDA_SUFFIX
ARG PYG_TORCH_VERSION
ARG CUDA_ARCH_LIST

# Fail fast if required build args are missing
RUN if [ -z "$TORCH_VERSION" ] || [ -z "$CUDA_SUFFIX" ] || [ -z "$PYG_TORCH_VERSION" ] || [ -z "$CUDA_ARCH_LIST" ]; then \
      echo "ERROR: Missing required build args. Use docker-compose.yml to build:" >&2; \
      echo "  docker compose -f docker/docker-compose.yml build cossad-cu124" >&2; \
      echo "  docker compose -f docker/docker-compose.yml build cossad-cu128" >&2; \
      exit 1; \
    fi

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl bzip2 ca-certificates \
    build-essential \
    ninja-build \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    gosu mc git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /app

# ---------------------------------------------------------------------------
# Create non-root user (matches host UID/GID to avoid permission issues)
# ---------------------------------------------------------------------------
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID appuser 2>/dev/null || true && \
    useradd -u $UID -g $GID -m -s /bin/bash appuser

# ---------------------------------------------------------------------------
# Install micromamba
# ---------------------------------------------------------------------------
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

ENV MAMBA_ROOT_PREFIX=/opt/conda

# ---------------------------------------------------------------------------
# Create conda environment
# ---------------------------------------------------------------------------
COPY docker/environment.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml && \
    micromamba clean --all --yes && \
    rm /tmp/environment.yml

# Set up shell for interactive sessions
RUN echo 'eval "$(micromamba shell hook -s bash)"' >> /root/.bashrc && \
    echo 'micromamba activate cossad' >> /root/.bashrc && \
    echo 'eval "$(micromamba shell hook -s bash)"' >> /home/appuser/.bashrc && \
    echo 'micromamba activate cossad' >> /home/appuser/.bashrc

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# CUDA architectures for JIT compilation
# cu124: Turing (7.5), Ampere (8.0, 8.6), Ada (8.9), Hopper (9.0)
# cu128: Ampere (8.0, 8.6), Ada (8.9), Hopper (9.0), Blackwell (10.0, 12.0)
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST}

# ---------------------------------------------------------------------------
# Install PyTorch + CUDA
# ---------------------------------------------------------------------------
RUN micromamba run -n cossad pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/${CUDA_SUFFIX}

# Install torch-geometric and CUDA extensions
RUN micromamba run -n cossad pip install --no-cache-dir \
    torch-geometric \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-${PYG_TORCH_VERSION}+${CUDA_SUFFIX}.html

# Install PyTorch-dependent packages
RUN micromamba run -n cossad pip install --no-cache-dir \
    kornia \
    timm \
    torchinfo \
    narwhals \
    pytorch-metric-learning

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
WORKDIR /app
COPY . /app/

VOLUME ["/data", "/weights", "/app"]

ENV POINTOPS_CACHE=/app/.cache/pointops

COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh
COPY --chmod=755 docker/initialize.sh /initialize.sh

EXPOSE 5678 6006

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
