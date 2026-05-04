#!/bin/bash
set -e

# ==============================================================================
# COSSAD Initialization Script (runs as appuser)
# ==============================================================================
#
# This script is called by entrypoint.sh after UID/GID mapping is complete.
# It runs as appuser to ensure proper file ownership and permissions.
#
# ==============================================================================

# ---------------------------------------------------------------------------
# Activate micromamba environment
# ---------------------------------------------------------------------------
eval "$(micromamba shell hook -s bash)"
micromamba activate cossad

# ---------------------------------------------------------------------------
# Git identity for DVC experiment tracking (local commits only, no push)
# ---------------------------------------------------------------------------
git config --global user.name  "COSSAD Docker" 2>/dev/null || true
git config --global user.email "docker@cossad.local" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Compile pointops CUDA extension if needed
# ---------------------------------------------------------------------------
POINTOPS_DIR="/app/external/riconv2/models/pointops"
POINTOPS_MARKER="${POINTOPS_CACHE:-/app/.cache/pointops}/.compiled"

if [ ! -f "$POINTOPS_MARKER" ]; then
    echo "=== Compiling pointops CUDA extension ==="

    if [ ! -d "$POINTOPS_DIR" ]; then
        echo "ERROR: pointops source not found at $POINTOPS_DIR"
        echo "Make sure the code is properly mounted or copied."
        exit 1
    fi

    cd "$POINTOPS_DIR"

    # Clean previous build artifacts
    rm -rf build/ dist/ *.egg-info/ src/*.so src/build/ *.so

    # Compile pointops via the pip install in dev mode.
    pip install -e . --no-build-isolation -v

    if [ $? -eq 0 ]; then
        mkdir -p "$(dirname $POINTOPS_MARKER)"
        touch "$POINTOPS_MARKER"
        echo "=== pointops compiled successfully ==="
    else
        echo "ERROR: pointops compilation failed"
        exit 1
    fi

    cd /app
else
    echo "=== pointops already compiled (use 'rm ${POINTOPS_MARKER}' to force recompile) ==="
fi

# ---------------------------------------------------------------------------
# Execute the user command
# ---------------------------------------------------------------------------
if [ "$COSSAD_DEBUG" = "1" ]; then
    echo "=== Starting in debug mode (debugpy on port 5678) ==="
    exec python -m debugpy --listen 0.0.0.0:5678 --wait-for-client "$@"
else
    exec "$@"
fi
