#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────── #
# Nuitka Build Script for GRAVITAS Engine                                      #
# ─────────────────────────────────────────────────────────────────────────── #
#
# Compiles gravitas and gravitas_engine packages into native .so modules
# for improved runtime performance.
#
# Prerequisites:
#   pip install nuitka ordered-set
#   apt install patchelf   # (only needed for --standalone builds)
#   apt install ccache     # (optional, speeds up recompilation)
#
# Usage:
#   ./build_nuitka.sh              # Compile both packages as modules
#   ./build_nuitka.sh --clean      # Remove build artifacts
#
# NOTE: Python 3.14 is experimentally supported by Nuitka 4.0.x.
#       Module compilation succeeds, but runtime loading may segfault.
#       Use Python 3.12/3.13 for production builds until Nuitka adds
#       full 3.14 support.
# ─────────────────────────────────────────────────────────────────────────── #

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(which python3)}"
OUTPUT_DIR="build"

if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning build artifacts..."
    rm -rf "$OUTPUT_DIR" *.so
    echo "Done."
    exit 0
fi

mkdir -p "$OUTPUT_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  GRAVITAS Engine — Nuitka Build"
echo "  Python: $($PYTHON --version 2>&1)"
echo "  Nuitka: $($PYTHON -m nuitka --version 2>&1 | head -1)"
echo "═══════════════════════════════════════════════════════════"
echo

# ── Compile gravitas (orchestration + plugins) ──────────────────────────── #
echo "[1/3] Compiling gravitas package (engine + plugins)..."
$PYTHON -m nuitka \
    --module gravitas \
    --include-package=gravitas \
    --output-dir="$OUTPUT_DIR" \
    2>&1 | tail -3
echo

# ── Compile gravitas_engine (core simulation) ───────────────────────────── #
echo "[2/3] Compiling gravitas_engine package (dynamics + env)..."
$PYTHON -m nuitka \
    --module gravitas_engine \
    --include-package=gravitas_engine \
    --output-dir="$OUTPUT_DIR" \
    2>&1 | tail -3
echo

# ── Compile extensions (military + political) ───────────────────────────── #
if [ -d "extensions" ]; then
    echo "[3/3] Compiling extensions package (military + political)..."
    $PYTHON -m nuitka \
        --module extensions \
        --include-package=extensions \
        --output-dir="$OUTPUT_DIR" \
        2>&1 | tail -3
    echo
else
    echo "[3/3] Skipping extensions (not found)"
fi

# ── Summary ─────────────────────────────────────────────────────────────── #
echo "═══════════════════════════════════════════════════════════"
echo "  Build complete. Compiled modules:"
ls -lh "$OUTPUT_DIR"/*.so 2>/dev/null || echo "  (no .so files found)"
echo "═══════════════════════════════════════════════════════════"
echo
echo "To use compiled modules, add build/ to PYTHONPATH:"
echo "  PYTHONPATH=$OUTPUT_DIR:\$PYTHONPATH python cli.py run moscow"
echo "  PYTHONPATH=$OUTPUT_DIR:\$PYTHONPATH python tests/train_moscow_selfplay.py"
