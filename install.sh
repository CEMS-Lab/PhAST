#!/bin/bash
# PhAST — Smart one-command installation
#
# Auto-detects hardware, installs the fastest available solver stack,
# and always tracks GitHub's latest version.
#
# Usage:
#   bash install.sh                           # Auto-detect everything
#   bash install.sh cuda                      # Force CUDA
#   bash install.sh cuda --cuda-version 12.4  # Force specific CUDA
#   bash install.sh mps                       # macOS Apple Silicon
#   bash install.sh cpu                       # CPU only
#   bash install.sh cpu --no-direct           # Skip PETSc/MUMPS attempt
#
# What it does:
#   1. Detects platform (CUDA / MPS / CPU) and GPU model
#   2. Installs PyTorch + PyG with correct wheels
#   3. Installs PhAST from this checkout in editable mode
#   4. Auto-installs best preconditioner libraries for your hardware:
#        A100/H100 → pyamgx (AmgX) + pyamg + pymetis
#        RTX/Quadro → pyamg + pymetis
#        macOS MPS  → (none needed, GMG is matrix-free)
#        CPU        → pyamg
#   5. Verifies everything works

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NEURAL_OP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLATFORM="auto"
CUDA_VER=""
TRY_DIRECT=1
REPO_URL="https://github.com/CEMS-Lab/PhAST.git"

while [[ $# -gt 0 ]]; do
    case $1 in
        auto|cuda|mps|cpu) PLATFORM="$1"; shift ;;
        --cuda-version) CUDA_VER="$2"; shift 2 ;;
        --no-direct) TRY_DIRECT=0; shift ;;
        *) echo "Unknown install option: $1"; exit 1 ;;
    esac
done

# ============================================================
# 1. AUTO-DETECT PLATFORM
# ============================================================
if [ "$PLATFORM" = "auto" ]; then
    if command -v nvidia-smi &>/dev/null; then
        PLATFORM="cuda"
    elif python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        PLATFORM="cuda"
    elif [ "$(uname)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        PLATFORM="mps"
    elif python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
        PLATFORM="mps"
    else
        PLATFORM="cpu"
    fi
fi

# Auto-detect CUDA version
if [ "$PLATFORM" = "cuda" ] && [ -z "$CUDA_VER" ]; then
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    elif command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    fi
    [ -z "$CUDA_VER" ] && CUDA_VER="12.8"
fi

# Detect GPU model for preconditioner selection
GPU_NAME=""
GPU_TIER="cpu"
if [ "$PLATFORM" = "cuda" ]; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
    GPU_NAME_LOWER=$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]')
    if echo "$GPU_NAME_LOWER" | grep -qE "a100|h100|h200|v100|a30"; then
        GPU_TIER="hpc"
    elif echo "$GPU_NAME_LOWER" | grep -qE "a2000|a4000|a5000|a6000|rtx|quadro|l40"; then
        GPU_TIER="workstation"
    else
        GPU_TIER="consumer"
    fi
elif [ "$PLATFORM" = "mps" ]; then
    GPU_TIER="mps"
    GPU_NAME="Apple Silicon"
fi

# Map CUDA version to PyTorch wheel suffix
cuda_wheel_suffix() {
    local ver="$1"
    local major="${ver%%.*}"
    local minor="${ver#*.}"
    minor="${minor%%.*}"
    if [ "$major" -ge 13 ] 2>/dev/null; then echo "cu128"
    elif [ "$major" = "12" ]; then
        if [ "$minor" -ge 8 ]; then echo "cu128"
        elif [ "$minor" -ge 6 ]; then echo "cu126"
        elif [ "$minor" -ge 4 ]; then echo "cu124"
        else echo "cu121"; fi
    elif [ "$major" = "11" ] && [ "$minor" -ge 8 ]; then echo "cu118"
    else echo "cu121"; fi
}

install_sparse_direct_optional() {
    if [ "$TRY_DIRECT" = "0" ]; then
        echo "  ├─ Skipping PETSc/MUMPS sparse-direct install (--no-direct)"
        return
    fi

    echo "  ├─ Checking PETSc/MUMPS sparse-direct backend..."
    if python3 -c "from phast.sparse_solve import available_sparse_backends; import sys; sys.exit(0 if available_sparse_backends().petsc else 1)" 2>/dev/null; then
        echo "  │  ✓ PETSc/MUMPS already functional"
        echo "  │  Validate with: python3 -m phast doctor"
        return
    fi

    if command -v mamba &>/dev/null; then
        echo "  │  Trying conda-forge PETSc/MUMPS with mamba..."
        mamba install -c conda-forge -y petsc petsc4py mumps-mpi 2>/dev/null && \
            echo "  │  ✓ PETSc/MUMPS packages installed" || \
            echo "  │  ✗ mamba PETSc/MUMPS install failed"
    elif command -v conda &>/dev/null; then
        echo "  │  Trying conda-forge PETSc/MUMPS with conda..."
        conda install -c conda-forge -y petsc petsc4py mumps-mpi 2>/dev/null && \
            echo "  │  ✓ PETSc/MUMPS packages installed" || \
            echo "  │  ✗ conda PETSc/MUMPS install failed"
    else
        echo "  │  No conda/mamba found; skipping automatic PETSc/MUMPS install"
        echo "  │  To enable later: mamba install -c conda-forge petsc petsc4py mumps-mpi"
    fi

    if python3 -c "from phast.sparse_solve import available_sparse_backends; import sys; sys.exit(0 if available_sparse_backends().petsc else 1)" 2>/dev/null; then
        echo "  │  ✓ PETSc/MUMPS smoke test passed"
    else
        echo "  │  ✗ PETSc/MUMPS smoke test did not pass; backend='auto' will fall back"
    fi
    echo "  │  Validate with: python3 -m phast doctor"
}

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              PhAST — Smart Installer                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Platform:     $PLATFORM"
[ -n "$GPU_NAME" ] && echo "  GPU:          $GPU_NAME ($GPU_TIER)"
[ "$PLATFORM" = "cuda" ] && echo "  CUDA:         $CUDA_VER"
echo ""

# ============================================================
# 2. PULL LATEST FROM GITHUB
# ============================================================
echo "── Syncing with GitHub (latest) ──────────────────────────"
if [ -d "${SCRIPT_DIR}/.git" ]; then
    cd "$SCRIPT_DIR"
    CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "main")
    echo "  Pulling latest from origin/${CURRENT_BRANCH}..."
    git pull origin "${CURRENT_BRANCH}" --ff-only 2>/dev/null || \
        echo "  ⚠ Could not pull (local changes?). Continuing with current version."
    cd "$NEURAL_OP_DIR"
else
    echo "  Not a git repo. To track latest:"
    echo "    git clone ${REPO_URL}"
fi
echo ""

# ============================================================
# 3. INSTALL PYTORCH + PYG
# ============================================================
TORCH_VERSION="2.8.0"

echo "── Installing PyTorch ${TORCH_VERSION} ───────────────────"
case "$PLATFORM" in
    cuda)
        CU_SUFFIX=$(cuda_wheel_suffix "$CUDA_VER")
        echo "  Wheels: ${CU_SUFFIX}"
        pip install torch==${TORCH_VERSION} torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/${CU_SUFFIX}
        echo ""
        echo "── Installing PyG (${CU_SUFFIX}) ─────────────────────────"
        pip install torch-scatter torch-sparse torch-cluster pyg-lib \
            -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CU_SUFFIX}.html
        ;;
    mps)
        pip install torch==${TORCH_VERSION} torchvision torchaudio
        echo ""
        echo "── Installing PyG (CPU wheels for macOS) ──────────────"
        pip install torch-scatter torch-sparse torch-cluster \
            -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html
        ;;
    cpu)
        pip install torch==${TORCH_VERSION} torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cpu
        echo ""
        echo "── Installing PyG (CPU) ──────────────────────────────"
        pip install torch-scatter torch-sparse torch-cluster \
            -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html
        ;;
    *)
        echo "Unknown platform: $PLATFORM"
        exit 1
        ;;
esac
pip install torch_geometric
echo ""

# ============================================================
# 4. INSTALL PhAST + BEST PRECONDITIONER STACK
# ============================================================
echo "── Installing PhAST ───────────────────────────────────────"
pip install -e "${SCRIPT_DIR}"
echo ""

echo "── Selecting optimal solver libraries for ${GPU_TIER} ─────"
echo ""

case "$GPU_TIER" in
    hpc)
        echo "  ┌─ HPC GPU detected (${GPU_NAME})"
        echo "  │  Optimal: AmgX (CUDA-native) → AMG (PyAMG) → GMG → Jacobi"
        echo "  │"

        # Always install PyAMG (safe, fast fallback)
        echo "  ├─ Installing pyamg (AMG hierarchy builder)..."
        pip install "pyamg>=5.0" && echo "  │  ✓ pyamg installed" || echo "  │  ✗ pyamg failed (AMG unavailable, GMG still works)"

        # Try pyamgx (requires AMGX C library pre-built)
        echo "  ├─ Installing pyamgx (NVIDIA AmgX, fastest)..."
        if [ -n "$AMGX_DIR" ] && [ -d "$AMGX_DIR" ]; then
            pip install cython 2>/dev/null
            cd /tmp && rm -rf pyamgx
            git clone https://github.com/shwina/pyamgx.git 2>/dev/null
            cd pyamgx && pip install . 2>/dev/null && echo "  │  ✓ pyamgx installed" || echo "  │  ✗ pyamgx build failed"
            cd "$SCRIPT_DIR"
        else
            echo "  │  ✗ AMGX_DIR not set — AmgX C library not found"
            echo "  │    To build: git clone --recursive https://github.com/NVIDIA/AMGX"
            echo "  │              cd AMGX && mkdir build && cd build"
            echo "  │              cmake .. -DCUDA_ARCH=80 && make -j8"
            echo "  │              export AMGX_DIR=\$PWD/.."
            echo "  │    Then re-run install.sh. Using PyAMG + CuPy instead."
        fi

        # CuPy for GPU-accelerated direct solver (--direct mode)
        echo "  ├─ Installing cupy (GPU sparse direct solver)..."
        # Detect CUDA major version for correct cupy wheel
        CUDA_MAJOR="${CUDA_VER%%.*}"
        if [ "$CUDA_MAJOR" -ge 12 ] 2>/dev/null; then
            pip install "cupy-cuda12x>=13.0" 2>/dev/null && echo "  │  ✓ cupy-cuda12x installed" || echo "  │  ✗ cupy failed (--direct will use CPU scipy)"
        elif [ "$CUDA_MAJOR" = "11" ] 2>/dev/null; then
            pip install "cupy-cuda11x>=13.0" 2>/dev/null && echo "  │  ✓ cupy-cuda11x installed" || echo "  │  ✗ cupy failed (--direct will use CPU scipy)"
        else
            echo "  │  ✗ cupy skipped (unknown CUDA ${CUDA_VER})"
        fi

        # METIS for multi-GPU partitioning
        echo "  ├─ Installing pymetis (mesh partitioning)..."
        pip install "pymetis>=2020.1" 2>/dev/null && echo "  │  ✓ pymetis installed" || echo "  │  ✗ pymetis failed (spatial bisection will be used)"

        echo "  └─ Done"
        ;;

    workstation)
        echo "  ┌─ Workstation GPU detected (${GPU_NAME})"
        echo "  │  Optimal: AmgX (CUDA-native) → AMG (PyAMG) → GMG → Jacobi"
        echo "  │"

        echo "  ├─ Installing pyamg (AMG hierarchy builder)..."
        pip install "pyamg>=5.0" && echo "  │  ✓ pyamg installed" || echo "  │  ✗ pyamg failed (GMG still works)"

        echo "  ├─ Installing pyamgx (NVIDIA AmgX, fastest)..."
        if pip install "pyamgx>=0.1" 2>/dev/null; then
            echo "  │  ✓ pyamgx installed — AmgX will be used (fastest)"
        else
            echo "  │  ✗ pyamgx failed — AmgX C libs not found"
            echo "  │    To enable: module load amgx && pip install pyamgx"
            echo "  │    Solver will use AMG (PyAMG + torch V-cycle) instead"
        fi

        echo "  ├─ Installing pymetis..."
        pip install "pymetis>=2020.1" 2>/dev/null && echo "  │  ✓ pymetis installed" || echo "  │  ✗ pymetis failed (spatial bisection will be used)"

        echo "  └─ Done"
        ;;

    mps)
        echo "  ┌─ Apple Silicon detected"
        echo "  │  Optimal: GMG (matrix-free, no extra deps needed)"
        echo "  │"
        echo "  ├─ Installing pyamg (optional, for CPU-side AMG)..."
        pip install "pyamg>=5.0" && echo "  │  ✓ pyamg installed (AMG available on CPU)" || echo "  │  ✗ pyamg failed (GMG is default, no problem)"
        echo "  └─ Done"
        ;;

    consumer)
        echo "  ┌─ CUDA GPU detected (${GPU_NAME})"
        echo "  │  Optimal: AmgX → AMG → GMG → Jacobi"
        echo "  │"

        echo "  ├─ Installing pyamg..."
        pip install "pyamg>=5.0" && echo "  │  ✓ pyamg installed" || echo "  │  ✗ pyamg failed (GMG still works)"

        echo "  ├─ Installing pyamgx (NVIDIA AmgX)..."
        if pip install "pyamgx>=0.1" 2>/dev/null; then
            echo "  │  ✓ pyamgx installed — AmgX will be used (fastest)"
        else
            echo "  │  ✗ pyamgx failed — solver will use AMG or GMG instead"
        fi

        echo "  └─ Done"
        ;;

    cpu|*)
        echo "  ┌─ CPU detected"
        echo "  │  Optimal: AMG (PyAMG, no transfer overhead) → GMG → Jacobi"
        echo "  │"
        echo "  ├─ Installing pyamg..."
        pip install "pyamg>=5.0" && echo "  │  ✓ pyamg installed" || echo "  │  ✗ pyamg failed (GMG still works)"
        echo "  └─ Done"
        ;;
esac

echo ""
echo "── Sparse-direct backend for implicit/QS workflows ─────────"
install_sparse_direct_optional

echo ""
echo "── Visualisation and trajectory workflow packages ──────────"
echo "  ├─ Zarr is installed as a core dependency for trajectory stores"
echo "  ├─ Installing fast PyVista/zstd writer (optional)..."
pip install "pyvista>=0.48" "pyvista-zstd>=0.2" 2>/dev/null && \
    echo "  │  ✓ fast .pv visualisation writer available" || \
    echo "  │  ✗ fast .pv writer unavailable; VTU/MP4/raster paths remain available"

# ============================================================
# 5. LINUX SYSTEM DEPS (gmsh needs libGL)
# ============================================================
if [ "$(uname)" = "Linux" ]; then
    echo ""
    echo "── Checking Linux system dependencies ─────────────────"
    if ! python3 -c "import gmsh; gmsh.initialize(); gmsh.finalize()" 2>/dev/null; then
        echo "  gmsh needs libGL. Attempting to install..."
        if command -v apt-get &>/dev/null; then
            sudo apt-get update -qq && sudo apt-get install -y -qq libgl1-mesa-glx libglu1-mesa 2>/dev/null || \
                echo "  ⚠ Could not install libGL. Run: sudo apt-get install libgl1-mesa-glx"
        elif command -v yum &>/dev/null; then
            sudo yum install -y mesa-libGL mesa-libGLU 2>/dev/null || \
                echo "  ⚠ Could not install libGL. Run: sudo yum install mesa-libGL"
        fi
    fi
fi

# ============================================================
# 6. PYTHONPATH SETUP
# ============================================================
echo ""
echo "── Setting up PYTHONPATH ──────────────────────────────────"
if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ] && [ "$(basename "$SHELL")" = "zsh" ]; then
    SHELL_RC="$HOME/.zshrc"
else
    SHELL_RC="$HOME/.bashrc"
fi

MARKER="# PhAST PYTHONPATH"
if grep -qF "$MARKER" "$SHELL_RC" 2>/dev/null; then
    echo "  Already configured in $SHELL_RC"
else
    echo "" >> "$SHELL_RC"
    echo "$MARKER" >> "$SHELL_RC"
    echo "export PYTHONPATH=\"${NEURAL_OP_DIR}:\${PYTHONPATH}\"" >> "$SHELL_RC"
    echo "  Added to $SHELL_RC"
fi
export PYTHONPATH="${NEURAL_OP_DIR}:${PYTHONPATH}"

# ============================================================
# 7. VERIFICATION
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    Verification                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

python3 -c "
import sys, os
sys.path.insert(0, '${NEURAL_OP_DIR}')

import torch
print(f'  PyTorch        {torch.__version__}')
if torch.cuda.is_available():
    print(f'  CUDA device    {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version   {torch.version.cuda}')
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'  MPS            available')

try:
    import torch_geometric
    print(f'  PyG            {torch_geometric.__version__}')
except ImportError:
    print(f'  PyG            MISSING')

# Core deps
for mod in ['numpy', 'scipy', 'matplotlib', 'h5py', 'meshio', 'zarr', 'numcodecs']:
    try:
        m = __import__(mod)
        print(f'  {mod:<13s}  {getattr(m, \"__version__\", \"ok\")}')
    except ImportError:
        print(f'  {mod:<13s}  MISSING')

try:
    import pyvista
    print(f'  pyvista        {pyvista.__version__}')
except ImportError:
    print(f'  pyvista        optional missing')

# Solver
try:
    from phast import FEMMesh, PhaseFieldDamageSolver, get_device_tier
    print(f'  phast OK')
except Exception as e:
    print(f'  phast FAILED ({e})')

# Preconditioner stack
print()
print('  Preconditioner availability:')
try:
    import pyamgx
    print(f'    AmgX (pyamgx)    ✓ — CUDA-native (fastest)')
except ImportError:
    print(f'    AmgX (pyamgx)    ✗ — not installed (optional)')

try:
    import pyamg
    print(f'    AMG  (pyamg)     ✓ v{pyamg.__version__} — GPU-native V-cycle')
except ImportError:
    print(f'    AMG  (pyamg)     ✗ — not installed')

try:
    import cupy
    print(f'    CuPy (direct)    ✓ v{cupy.__version__} — GPU sparse LU (--direct)')
except ImportError:
    print(f'    CuPy (direct)    ✗ — not installed (--direct uses CPU scipy)')

print(f'    GMG              ✓ — built-in (matrix-free)')
print(f'    Jacobi           ✓ — built-in (baseline)')

# Show what auto will pick
try:
    from phast.device import get_device_tier, detect_device
    dev = detect_device()
    tier = get_device_tier(dev)
    print()
    print(f'  Device tier: {tier[\"tier\"]} ({tier[\"name\"]})')
    print(f'  Auto will try: ', end='')
    if tier['type'] == 'cuda':
        print('amgx → amg → gmg → jacobi')
    elif tier['type'] == 'cpu':
        print('amg → gmg → jacobi')
    else:
        print('gmg → jacobi')
except Exception:
    pass
"

python3 -m phast doctor || true

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              Installation complete!                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Quick start:"
echo "    cd ${NEURAL_OP_DIR}"
echo "    python -m phast run examples/quasistatic/miehe_tension/config.yaml --validate-only"
echo ""
echo "  Update to latest:"
echo "    cd ${SCRIPT_DIR} && git pull && pip install -e ."
echo ""
echo "  NOTE: Open a new terminal or run 'source $SHELL_RC' to activate PYTHONPATH."
echo ""
