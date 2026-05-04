#!/bin/bash
# COSSAD Docker Build Script
set -e

# A full path to the script, no matter where you call it from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# A full path to the project root.
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

usage() {
    cat << EOF
Usage: $0 [OPTIONS] VARIANT

Build COSSAD Docker images.

VARIANTS:
    cu124       CUDA 12.4 + PyTorch 2.5
    cu128       CUDA 12.8 + PyTorch 2.8
    all         Build all variants

OPTIONS:
    -n, --no-cache  Build without cache
    -h, --help      Show this help

EXAMPLES:
    $0 cu124            # Build CUDA 12.4 variant
    $0 cu128            # Build CUDA 12.8 variant
    $0 all              # Build everything
EOF
}

NO_CACHE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--no-cache) NO_CACHE="--no-cache"; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             VARIANT="$1"; shift ;;
    esac
done

if [ -z "$VARIANT" ]; then
    echo "ERROR: No variant specified"
    usage
    exit 1
fi

build_variant() {
    local variant=$1

    echo "=== Building cossad:$variant ==="
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" build $NO_CACHE "cossad-$variant"
    echo "=== cossad:$variant built successfully ==="
}

# Remove the "pointops compiled" flag
rm -f "$PROJECT_ROOT/.cache/pointops/.compiled"

case $VARIANT in
    cu124|cu128)
        build_variant "$VARIANT"
        ;;
    all)
        build_variant "cu124"
        build_variant "cu128"
        ;;
    *)
        echo "ERROR: Unknown variant: $VARIANT"
        usage
        exit 1
        ;;
esac

echo ""
echo "Build complete!"
