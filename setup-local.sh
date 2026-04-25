#!/bin/bash
# setup-local.sh
#
# Sets up a local Python venv and builds the C++ poker-sim with libtorch.
# Run once from the repo root before using the training loop locally.
#
# Prerequisites (macOS):
#   brew install cmake
#   brew install python@3.12  (or any Python 3.10+)
#
# Usage:
#   ./setup-local.sh
#   source .venv/bin/activate          # activate venv for python commands
#   ./build/poker/poker-sim weights/ --mode seqcomm --episodes 100

set -e

PYTHON="${PYTHON:-python3.12}"

# ── 1. Create venv ────────────────────────────────────────────────────────────
if [ ! -f ".venv/bin/python3" ]; then
    echo "Creating .venv with $PYTHON …"
    $PYTHON -m venv .venv
fi

# ── 2. Install Python dependencies ───────────────────────────────────────────
echo "Installing Python dependencies …"
.venv/bin/pip install --quiet \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ── 3. Configure cmake with libtorch ─────────────────────────────────────────
TORCH_PREFIX=$(.venv/bin/python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
echo "libtorch prefix: $TORCH_PREFIX"

cmake -B build \
    -DUSE_TORCH=ON \
    -DCMAKE_PREFIX_PATH="$TORCH_PREFIX"

# ── 4. Build poker-sim ────────────────────────────────────────────────────────
echo "Building poker-sim …"
cmake --build build --target poker-sim

echo ""
echo "Done. Run:"
echo "  # terminal 1 — C++ simulation"
echo "  ./build/poker/poker-sim weights/ --mode seqcomm --episodes 2000"
echo ""
echo "  # terminal 2 — Python weight updates"
echo "  .venv/bin/python3 -m training.train_from_cpp weights/"
echo ""
echo "  # Python training (no C++ sync)"
echo "  .venv/bin/python3 -m training.train --env poker --episodes 500"
