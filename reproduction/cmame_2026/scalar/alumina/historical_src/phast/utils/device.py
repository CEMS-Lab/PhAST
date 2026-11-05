"""
Device management, AMP (mixed precision), and profiling for phast.

Provides:
  - Auto-detection of best available device (CUDA > MPS > CPU)
  - AMP context managers with accuracy fallbacks
  - Profiling timer for hot-path benchmarking
  - Device capability queries (float64 support, compile support, etc.)

Usage::

    from phast.device import DeviceContext

    ctx = DeviceContext()            # auto-detect best device
    ctx = DeviceContext('cuda:0')    # explicit
    ctx = DeviceContext(amp=True)    # enable mixed precision

    mesh = FEMMesh(path, device=ctx.device, dtype=ctx.dtype)
    with ctx.amp_context():
        f_int = fem.internal_force(u, d)

    ctx.profiler.summary()           # print timing breakdown
"""

import torch
import time
import subprocess
from contextlib import contextmanager
from typing import Dict, Optional


def _query_gpu_utilization() -> Dict[int, float]:
    """Query GPU utilization (%) via nvidia-smi. Returns {gpu_idx: util%}."""
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=index,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return {}
        utils = {}
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            if len(parts) == 2:
                idx, util = int(parts[0].strip()), float(parts[1].strip())
                utils[idx] = util
        return utils
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return {}


def _cuda_gpu_with_most_free_memory() -> int:
    """Return the CUDA device index with the best combined score.

    Score = free_vram_gb * (1 - utilization / 100).
    Falls back to free VRAM only if nvidia-smi is unavailable.
    """
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        return 0

    gpu_utils = _query_gpu_utilization()
    has_util = len(gpu_utils) > 0

    best_idx, best_score = 0, -1.0
    for i in range(n_gpus):
        free, total = torch.cuda.mem_get_info(i)
        free_gb = free / (1024**3)
        props = torch.cuda.get_device_properties(i)

        util = gpu_utils.get(i, 0.0)
        if has_util:
            score = free_gb * (1.0 - util / 100.0)
            print(f"[device] GPU {i}: {props.name}, "
                  f"{free_gb:.1f} / {total / (1024**3):.1f} GB free, "
                  f"util {util:.0f}%, score {score:.2f}", flush=True)
        else:
            score = free_gb
            print(f"[device] GPU {i}: {props.name}, "
                  f"{free_gb:.1f} / {total / (1024**3):.1f} GB free", flush=True)

        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def detect_device(preferred: Optional[str] = None) -> torch.device:
    """Auto-detect the best available compute device.

    Priority: preferred (if available) > CUDA (most free VRAM) > MPS > CPU.

    When multiple CUDA GPUs are available and no specific GPU is requested
    (i.e. preferred is 'cuda' or None), the GPU with the most free memory
    is selected automatically.

    Parameters
    ----------
    preferred : str or None
        'cuda' (auto-select best GPU), 'cuda:N' (specific GPU),
        'mps', or 'cpu'. If the preferred device is unavailable,
        falls back to the next best option with a warning.
    """
    if preferred is not None:
        dev = torch.device(preferred)
        if dev.type == 'cuda' and torch.cuda.is_available():
            # 'cuda' without index -> pick GPU with most free memory
            if dev.index is None:
                idx = _cuda_gpu_with_most_free_memory()
                dev = torch.device(f'cuda:{idx}')
                print(f"[device] Auto-selected cuda:{idx} (most free VRAM)",
                      flush=True)
            return dev
        if dev.type == 'mps' and torch.backends.mps.is_available():
            # User explicitly requested MPS — honour it
            print(f"[device] MPS requested explicitly. Note: MPS lacks float64 "
                  f"support; damage solver falls back to CPU. For best "
                  f"performance use --device cpu.", flush=True)
            return dev
        if dev.type == 'cpu':
            return dev
        import warnings
        warnings.warn(f"Requested device '{preferred}' not available, falling back to auto-detect",
                      RuntimeWarning, stacklevel=2)

    if torch.cuda.is_available():
        idx = _cuda_gpu_with_most_free_memory()
        dev = torch.device(f'cuda:{idx}')
        print(f"[device] Auto-selected cuda:{idx} (most free VRAM)",
              flush=True)
        return dev
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS auto-detected but CPU is faster for phase-field (no float64 on
        # MPS means the CG damage solver ping-pongs tensors to CPU anyway).
        # Use CPU directly to avoid the transfer overhead.
        print(f"[device] MPS available but auto-selecting CPU instead "
              f"(MPS lacks float64; CPU avoids MPS↔CPU transfer overhead). "
              f"Use --device mps to force MPS.", flush=True)
        return torch.device('cpu')
    return torch.device('cpu')


def device_supports_float64(device: torch.device) -> bool:
    """Check if the device supports float64 operations."""
    if device.type == 'mps':
        return False  # MPS does not support float64
    return True


def device_supports_compile(device: torch.device) -> bool:
    """Check if torch.compile is supported on this device."""
    if device.type == 'mps':
        return False  # torch.compile has limited MPS support
    # torch.compile requires PyTorch >= 2.0
    try:
        major, minor = torch.__version__.split('.')[:2]
        return int(major) >= 2
    except (ValueError, IndexError):
        return False


def get_device_tier(device: torch.device) -> dict:
    """Query device capabilities for preconditioner selection.

    Returns a dict with:
      - 'type': 'cuda', 'mps', or 'cpu'
      - 'name': GPU name or 'cpu'
      - 'vram_gb': total VRAM in GB (0 for CPU)
      - 'compute_capability': (major, minor) tuple for CUDA, None otherwise
      - 'float64': whether device supports float64
      - 'tier': 'hpc' (A100/H100/V100), 'workstation' (RTX/Quadro),
                'consumer' (GeForce), 'mps', or 'cpu'
      - 'recommended_preconditioner': 'amg', 'gmg', or 'jacobi'
    """
    info = {
        'type': device.type,
        'name': 'cpu',
        'vram_gb': 0.0,
        'compute_capability': None,
        'float64': device_supports_float64(device),
        'tier': 'cpu',
        'recommended_preconditioner': 'gmg',
    }

    if device.type != 'cuda':
        if device.type == 'mps':
            info['name'] = 'Apple Silicon (MPS)'
            info['tier'] = 'mps'
            info['recommended_preconditioner'] = 'gmg'
        else:
            info['tier'] = 'cpu'
            info['recommended_preconditioner'] = 'amg'
        return info

    props = torch.cuda.get_device_properties(device)
    info['name'] = props.name
    info['vram_gb'] = round(props.total_memory / (1024**3), 1)
    info['compute_capability'] = (props.major, props.minor)

    name_lower = props.name.lower()
    # HPC: data-center GPUs (match full model tokens to avoid 'a30' matching 'a3000')
    import re
    tokens = re.split(r'[\s\-/]+', name_lower)
    hpc_models = {'a100', 'h100', 'h200', 'v100', 'a30', 'a40', 'a10',
                  'mi250', 'mi300', 'b100', 'b200', 'gb200'}
    if hpc_models & set(tokens):
        info['tier'] = 'hpc'
        info['recommended_preconditioner'] = 'amgx'
    elif any(k in name_lower for k in ('a2000', 'a4000', 'a5000', 'a6000',
                                        'a3000', 'a1000',
                                        'rtx 4', 'rtx 3', 'rtx 2',
                                        'quadro', 'l40')):
        info['tier'] = 'workstation'
        info['recommended_preconditioner'] = 'amgx'
    else:
        info['tier'] = 'consumer'
        info['recommended_preconditioner'] = 'amgx'

    return info


def estimate_vram_mb(n_nodes, n_elems, dtype=torch.float64,
                     use_multigrid=True, preconditioner='gmg'):
    """Estimate GPU VRAM usage for a given mesh size.

    The matrix-free solver has O(N) memory. Main allocations:
    - State: u(N,2), v(N,2), a(N,2), d(N), H(E), f_ext(N,2)
    - Mesh: nodes(N,2), elements(E,3), grad_phi(E,3,2), areas(E)
    - CG: 3-4 vectors of size N (r, z, p, Ap) for each solver
    - Multigrid: coarse matrix (~N/4 x N/4 dense) + aggregation

    Parameters
    ----------
    n_nodes : int
    n_elems : int
    dtype : torch.dtype
    use_multigrid : bool
    preconditioner : str

    Returns
    -------
    dict with 'total_mb', 'state_mb', 'mesh_mb', 'cg_mb', 'mg_mb'
    """
    bytes_per_float = 8 if dtype == torch.float64 else 4
    bpf = bytes_per_float

    # State variables
    state = (n_nodes * 2 * bpf * 3   # u, v, a: (N,2)
             + n_nodes * bpf          # d: (N,)
             + n_elems * bpf          # H_elem: (E,)
             + n_nodes * 2 * bpf)     # f_ext: (N,2)

    # Mesh data
    mesh = (n_nodes * 2 * bpf        # nodes: (N,2)
            + n_elems * 3 * 8         # elements: (E,3) int64
            + n_elems * 3 * 2 * bpf   # grad_phi: (E,3,2)
            + n_elems * bpf           # areas: (E,)
            + n_elems * 3 * 8         # elem_flat: (E*3) int64
            + n_nodes * bpf)          # M_scalar: (N,)

    # CG vectors (mechanics + damage)
    # Mechanics: u, r, z, p, Ap — each (N,2)
    # Damage: d, r, z, p, Ad — each (N,) but in float64
    cg_mech = n_nodes * 2 * bpf * 5
    cg_dmg = n_nodes * 8 * 5  # always float64
    cg = cg_mech + cg_dmg

    # Multigrid
    mg = 0
    if use_multigrid or preconditioner in ('gmg', 'amg', 'amgx', 'auto'):
        n_coarse = n_nodes // 4
        # Coarse matrix is sparse, not dense. Estimate ~10 nonzeros per row.
        mg = (n_coarse * 10 * bpf          # sparse coarse matrix
              + n_nodes * 8                # agg_id: (N,) int64
              + n_coarse * bpf)            # coarse vectors

    total = state + mesh + cg + mg

    return {
        'total_mb': total / 1e6,
        'state_mb': state / 1e6,
        'mesh_mb': mesh / 1e6,
        'cg_mb': cg / 1e6,
        'mg_mb': mg / 1e6,
        'n_nodes': n_nodes,
        'n_elems': n_elems,
    }


class Profiler:
    """Lightweight profiler for tracking time spent in solver phases.

    Records cumulative time and call counts per named region.
    No overhead when disabled.

    Usage::

        prof = Profiler(enabled=True)
        with prof.region('mechanics'):
            # ... mechanics solve ...
        with prof.region('damage'):
            # ... damage solve ...
        prof.summary()
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._timings = {}  # name -> [total_time, count]
        self._sync_fn = None

    def set_sync(self, device: torch.device):
        """Set synchronization function for GPU timing accuracy."""
        if device.type == 'cuda':
            self._sync_fn = torch.cuda.synchronize
        else:
            self._sync_fn = None

    @contextmanager
    def region(self, name: str):
        """Time a named code region."""
        if not self.enabled:
            yield
            return

        if self._sync_fn:
            self._sync_fn()
        t0 = time.perf_counter()
        yield
        if self._sync_fn:
            self._sync_fn()
        elapsed = time.perf_counter() - t0

        if name not in self._timings:
            self._timings[name] = [0.0, 0]
        self._timings[name][0] += elapsed
        self._timings[name][1] += 1

    def reset(self):
        """Clear all accumulated timings."""
        self._timings.clear()

    def summary(self) -> str:
        """Return a formatted summary of all timed regions."""
        if not self._timings:
            return "[Profiler] No timings recorded."

        total = sum(v[0] for v in self._timings.values())
        lines = ["[Profiler] Timing breakdown:"]
        lines.append(f"  {'Region':<25s} {'Total (s)':>10s} {'Calls':>8s} "
                     f"{'Avg (ms)':>10s} {'%':>6s}")
        lines.append("  " + "-" * 63)

        for name, (t, n) in sorted(self._timings.items(),
                                     key=lambda x: -x[1][0]):
            avg_ms = (t / n * 1000) if n > 0 else 0
            pct = (t / total * 100) if total > 0 else 0
            lines.append(f"  {name:<25s} {t:10.3f} {n:8d} "
                         f"{avg_ms:10.2f} {pct:5.1f}%")

        lines.append("  " + "-" * 63)
        lines.append(f"  {'TOTAL':<25s} {total:10.3f}")
        return '\n'.join(lines)


class DeviceContext:
    """Unified device, dtype, and profiling context.

    Parameters
    ----------
    device : str or None
        Device string ('cuda', 'mps', 'cpu') or None for auto-detect.
    amp : bool
        Deprecated, ignored. Kept for backward compatibility.
    dtype : torch.dtype or None
        Base dtype. None = auto (float64 if supported, else float32).
    profile : bool
        Enable built-in profiling.
    compile_solvers : bool or None
        Apply torch.compile to CG inner loops for JIT optimization.
        - True  : force ON (warn + disable if device does not support).
        - False : force OFF.
        - None  : auto-decide. If ``energy_split`` is passed, enable iff
          device is CUDA *and* split is amor/isotropic/volumetric_deviatoric
          (no eigenvalue-sign flips). If ``energy_split`` is not passed,
          fall back to OFF for backward compatibility.
    energy_split : str or None
        The material's energy split (``spectral``, ``amor``, ``isotropic``,
        ``volumetric_deviatoric``, ``star_convex``). Only used to decide
        the auto-compile policy when ``compile_solvers is None``. Pass the
        material's split string so the decision is correct.
    """

    def __init__(self, device: Optional[str] = None, amp: Optional[bool] = None,
                 dtype: Optional[torch.dtype] = None,
                 profile: bool = False,
                 compile_solvers: Optional[bool] = None,
                 energy_split: Optional[str] = None):

        self.device = detect_device(device)
        self.has_float64 = device_supports_float64(self.device)
        self.has_compile = device_supports_compile(self.device)

        # Base dtype: prefer float64 for accuracy, fall back on MPS
        if dtype is not None:
            self.dtype = dtype
        elif self.has_float64:
            self.dtype = torch.float64
        else:
            self.dtype = torch.float32
            print(f"[device] {self.device} does not support float64, "
                  f"using float32 (damage solver will use CPU float64 fallback)",
                  flush=True)

        # AMP removed: phase-field fracture requires float64 precision throughout.
        # The CG damage solver condition numbers (Gc*l0 vs Gc/l0 ~ 10^10) are
        # incompatible with reduced precision. AMP context managers were defined
        # but never used in the solver pipeline.
        self._amp_enabled = False

        # Compile config: torch.compile causes recompilation loops on
        # spectral split (eigenvalue-sign flips per element change the
        # computation graph). On amor / isotropic / volumetric_deviatoric
        # there is no such dynamic branching, so compile is a near-pure
        # win on CUDA. Policy:
        #   compile_solvers=True      : force ON (disable if unsupported)
        #   compile_solvers=False     : force OFF
        #   compile_solvers=None, split provided : auto-decide
        #   compile_solvers=None, no split       : OFF (backward compat)
        _SAFE_SPLITS_FOR_COMPILE = {'amor', 'isotropic', 'volumetric_deviatoric'}
        if compile_solvers is None:
            if (energy_split is not None
                and self.device.type == 'cuda'
                and self.has_compile
                and energy_split in _SAFE_SPLITS_FOR_COMPILE):
                self.compile_solvers = True
                print(f"[device] torch.compile auto-enabled "
                      f"(device=cuda, split={energy_split}). Pass "
                      f"--no-compile to override.", flush=True)
            else:
                self.compile_solvers = False
                if (energy_split is not None
                    and self.device.type == 'cuda'
                    and self.has_compile
                    and energy_split == 'spectral'):
                    print(f"[device] torch.compile auto-disabled "
                          f"(spectral split triggers recompilation loops). "
                          f"Pass --compile to override.", flush=True)
        else:
            self.compile_solvers = compile_solvers and self.has_compile
            if compile_solvers and not self.has_compile:
                print(f"[device] torch.compile not supported on {self.device}, "
                      f"disabled", flush=True)
            elif compile_solvers:
                print(f"[device] torch.compile enabled (explicit). "
                      f"First few steps may be slower due to compilation.",
                      flush=True)

        # Profiler
        self.profiler = Profiler(enabled=profile)
        self.profiler.set_sync(self.device)

        print(f"[device] Context: device={self.device}, dtype={self.dtype}, "
              f"compile={self.compile_solvers}, profile={profile}", flush=True)
        if self.device.type == 'cpu' and torch.cuda.is_available():
            print(f"[device] Hint: CUDA is available. Use --device cuda for "
                  f"3-8x speedup on meshes >10k nodes.", flush=True)
        if self.device.type == 'cuda' and not self.compile_solvers:
            print(f"[device] Hint: Use --compile for torch.compile JIT "
                  f"optimization (10-30% speedup after warmup).", flush=True)

    @property
    def amp_enabled(self) -> bool:
        return self._amp_enabled

    @contextmanager
    def amp_context(self):
        """No-op context manager (AMP removed — float64 required for PF fracture)."""
        yield

    @contextmanager
    def amp_context_off(self):
        """No-op context manager (AMP removed — float64 required for PF fracture)."""
        yield

    def cg_device_and_dtype(self):
        """Get the device and dtype for CG solvers.

        CG damage solver requires float64. On MPS this means CPU fallback.

        Returns
        -------
        cg_device : torch.device
        cg_dtype : torch.dtype
        """
        if self.has_float64:
            return self.device, torch.float64
        else:
            # MPS: fall back to CPU for float64 CG
            return torch.device('cpu'), torch.float64

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the context device (no-op if already there)."""
        if tensor.device == self.device:
            return tensor
        return tensor.to(self.device)

    def summary(self) -> str:
        lines = [
            f"DeviceContext:",
            f"  device     = {self.device}",
            f"  dtype      = {self.dtype}",
            f"  float64    = {self.has_float64}",
            f"  precision  = float64 (AMP removed)",
            f"  compile    = {self.compile_solvers}",
            f"  profiling  = {self.profiler.enabled}",
        ]
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            lines.append(f"  GPU        = {props.name}")
            lines.append(f"  VRAM       = {props.total_memory / (1024**3):.1f} GB")
        return '\n'.join(lines)

    def __repr__(self):
        return self.summary()
