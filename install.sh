#!/usr/bin/env bash
# PhAST source installer.
#
# Usage:
#   bash install.sh                 # install editable package
#   bash install.sh cpu             # force CPU PyTorch wheel
#   bash install.sh cuda            # install CUDA PyTorch wheel
#   bash install.sh cuda cu128      # install a specific CUDA wheel suffix
#   bash install.sh mps             # macOS/Apple Silicon default PyTorch wheel
#   bash install.sh --docs          # also install docs dependencies
#   bash install.sh --hpc           # also install pip-safe HPC extras

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM="auto"
CUDA_SUFFIX=""
INSTALL_DOCS=0
INSTALL_HPC=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        auto|cpu|cuda|mps)
            PLATFORM="$1"
            shift
            ;;
        cu118|cu121|cu124|cu126|cu128)
            CUDA_SUFFIX="$1"
            shift
            ;;
        --docs)
            INSTALL_DOCS=1
            shift
            ;;
        --hpc)
            INSTALL_HPC=1
            shift
            ;;
        -h|--help)
            sed -n '1,16p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown install option: $1" >&2
            exit 1
            ;;
    esac
done

python_bin="${PYTHON:-python3}"

if [[ "$PLATFORM" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        PLATFORM="cuda"
    elif [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
        PLATFORM="mps"
    else
        PLATFORM="cpu"
    fi
fi

if [[ "$PLATFORM" == "cuda" && -z "$CUDA_SUFFIX" ]]; then
    CUDA_SUFFIX="cu128"
fi

echo "PhAST source install"
echo "  repo:     $REPO_ROOT"
echo "  python:   $($python_bin -c 'import sys; print(sys.executable)')"
echo "  platform: $PLATFORM"
[[ -n "$CUDA_SUFFIX" ]] && echo "  torch:    $CUDA_SUFFIX"

"$python_bin" -m pip install --upgrade pip

case "$PLATFORM" in
    cuda)
        "$python_bin" -m pip install torch torchvision torchaudio \
            --index-url "https://download.pytorch.org/whl/${CUDA_SUFFIX}"
        ;;
    cpu)
        "$python_bin" -m pip install torch torchvision torchaudio \
            --index-url "https://download.pytorch.org/whl/cpu"
        ;;
    mps)
        "$python_bin" -m pip install torch torchvision torchaudio
        ;;
    *)
        echo "Unsupported platform: $PLATFORM" >&2
        exit 1
        ;;
esac

extras=()
if [[ "$INSTALL_HPC" == "1" ]]; then
    extras+=("hpc")
fi

if [[ "${#extras[@]}" -gt 0 ]]; then
    extra_spec="[$(IFS=,; echo "${extras[*]}")]"
else
    extra_spec=""
fi

"$python_bin" -m pip install -e "${REPO_ROOT}${extra_spec}"

if [[ "$INSTALL_DOCS" == "1" ]]; then
    "$python_bin" -m pip install -r "$REPO_ROOT/requirements-docs.txt"
fi

echo
echo "Verification"
"$python_bin" - <<'PY'
import torch
import phast

print(f"  torch: {torch.__version__}")
print(f"  cuda available: {torch.cuda.is_available()}")
if hasattr(torch.backends, "mps"):
    print(f"  mps available: {torch.backends.mps.is_available()}")
print(f"  phast: {phast.__name__}")
PY

"$python_bin" -m phast doctor

echo
echo "Install complete."
echo "Try: python -m phast run examples/quasistatic/miehe_tension/config.yaml --validate-only"
