#!/bin/bash
set -e

# ==============================================================================
# COSSAD Docker Entrypoint (runs as root)
# ==============================================================================
#
# This script runs as root to perform privileged operations that cannot be done
# as a regular user. After setup, it hands off to initialize.sh which runs as
# appuser.
#
# Why two scripts?
# ----------------
# 1. UID/GID remapping requires root (usermod/groupmod are privileged commands)
# 2. Pointops compilation and Python execution should run as appuser so that:
#    - Compiled files are owned by appuser (not root)
#    - File permissions on mounted volumes work correctly
#    - The container follows security best practices (principle of least privilege)
#
# Flow:
#   entrypoint.sh (root)     initialize.sh (appuser)
#   ┌──────────────────┐     ┌─────────────────────────┐
#   │ 1. Map UID/GID   │     │ 1. Activate micromamba  │
#   │ 2. Fix home dirs │────▶│ 2. Compile pointops     │
#   │ 3. exec gosu ... │     │ 3. exec user command    │
#   └──────────────────┘     └─────────────────────────┘
#
# ==============================================================================

# ---------------------------------------------------------------------------
# User mapping: adjust appuser UID/GID to match host user
# ---------------------------------------------------------------------------
TARGET_UID=${HOST_UID:-1000}
TARGET_GID=${HOST_GID:-1000}
CURRENT_UID=$(id -u appuser)
CURRENT_GID=$(id -g appuser)

if [ "$TARGET_GID" != "$CURRENT_GID" ]; then
    groupmod -g "$TARGET_GID" appuser 2>/dev/null || true
fi

if [ "$TARGET_UID" != "$CURRENT_UID" ]; then
    usermod -u "$TARGET_UID" appuser 2>/dev/null || true
fi

# Fix ownership of appuser's home directory
chown -R appuser:appuser /home/appuser

# ---------------------------------------------------------------------------
# Hand off to initialize.sh as appuser
# ---------------------------------------------------------------------------
exec gosu appuser /initialize.sh "$@"
