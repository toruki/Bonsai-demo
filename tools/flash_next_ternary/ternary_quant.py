"""Group-wise {-1,0,+1} quantizers.

Phase 1 established that the exact threshold / scale rule PrismML uses is NOT
recoverable from this repository or from the shipped GGUFs (see
docs/ternary_analysis.md §15). So rather than guessing one and calling it "Bonsai",
we implement four candidate rules and measure them side by side. Everything here is
**Bonsai-inspired**, not Bonsai.

All methods operate on groups of `group` consecutive elements along the last axis,
which is the same axis and group size (128) the PQ2_0 / PTQ1_0 block layout uses.

Returned `t` is int8 in {-1,0,+1}; `s` is one float32 scale per group.
The dequantized weight is `s_g * t_i`, which is exactly what the packers store
(their `d = amax(group)` recovers `s_g` when every `t` is ternary).
"""

from __future__ import annotations

import numpy as np


def _groups(w: np.ndarray, group: int) -> np.ndarray:
    w = np.ascontiguousarray(w, dtype=np.float32)
    if w.shape[-1] % group:
        raise ValueError(f"last dim {w.shape[-1]} not divisible by group {group}")
    return w.reshape(-1, group)


def _finish(t: np.ndarray, s: np.ndarray, shape: tuple[int, ...]
            ) -> tuple[np.ndarray, np.ndarray]:
    # a group that came out all-zero would store d=0 and dequantize to 0 anyway;
    # keep the scale at 0 so the packed form round-trips.
    s = np.where((t != 0).any(axis=1), s, 0.0).astype(np.float32)
    # PQ2_0 and PTQ1_0 both store one *fp16* scale per group, so anything finer than
    # fp16 is not representable. Round here rather than pretending otherwise.
    s = s.astype(np.float16).astype(np.float32)
    return t.astype(np.int8).reshape(shape), s


def _assign(g: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Round-to-nearest ternary assignment at scale s (threshold = s/2)."""
    inv = np.where(s > 0, 1.0 / np.where(s == 0, 1.0, s), 0.0)[:, None]
    return np.clip(np.rint(g * inv), -1.0, 1.0)


def _mse(g: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    return ((g - s[:, None] * t) ** 2).mean(axis=1)


# ------------------------------------------------------------------ A: absmean

def method_a(w: np.ndarray, group: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """s = mean(|w|) per group, then round-to-nearest.

    The BitNet b1.58 style rule. Threshold is implicitly s/2 = 0.5*mean|w|.
    """
    g = _groups(w, group)
    s = np.abs(g).mean(axis=1).astype(np.float32)
    return _finish(_assign(g, s), s, w.shape)


# ------------------------------------------------- B: 1-D MSE-optimal scale

def method_b(w: np.ndarray, group: int = 128, n_grid: int = 96,
             lo: float = 0.2, hi: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """s minimizing ||w - s*round_ternary(w/s)||^2, searched over s = r*mean|w|.

    The assignment stays plain round-to-nearest; only the scale moves. Grid search
    is used rather than a solver because the objective is piecewise-quadratic with
    kinks at every s = 2|w_i|, so gradient methods land on the wrong piece.
    """
    g = _groups(w, group)
    base = np.abs(g).mean(axis=1).astype(np.float32)
    base = np.where(base > 0, base, 1.0)

    best_s = base.copy()
    best_e = np.full(g.shape[0], np.inf, dtype=np.float32)
    for r in np.linspace(lo, hi, n_grid, dtype=np.float32):
        s = base * r
        e = _mse(g, _assign(g, s), s)
        upd = e < best_e
        best_e = np.where(upd, e, best_e)
        best_s = np.where(upd, s, best_s)
    return _finish(_assign(g, best_s), best_s, w.shape)


# ------------------------------------- C: threshold grid + closed-form scale

def method_c(w: np.ndarray, group: int = 128, n_grid: int = 96,
             lo: float = 0.1, hi: float = 1.6) -> tuple[np.ndarray, np.ndarray]:
    """Grid over the zero threshold; scale is the closed-form optimum for that support.

    For a fixed support S = {i : |w_i| > theta}, the MSE-optimal scale is
    s* = sum_{i in S} |w_i| / |S| (the TWN result; TWN then approximates the optimal
    theta as 0.7*mean|w| -- here we search it instead).
    """
    g = _groups(w, group)
    absg = np.abs(g)
    base = absg.mean(axis=1).astype(np.float32)
    base = np.where(base > 0, base, 1.0)
    sign = np.sign(g)

    best_t = np.zeros_like(g)
    best_s = np.zeros(g.shape[0], dtype=np.float32)
    best_e = np.full(g.shape[0], np.inf, dtype=np.float32)
    for r in np.linspace(lo, hi, n_grid, dtype=np.float32):
        theta = base * r
        keep = absg > theta[:, None]
        cnt = keep.sum(axis=1)
        ssum = np.where(keep, absg, 0.0).sum(axis=1)
        s = np.where(cnt > 0, ssum / np.maximum(cnt, 1), 0.0).astype(np.float32)
        t = sign * keep
        e = _mse(g, t, s)
        upd = e < best_e
        best_e = np.where(upd, e, best_e)
        best_s = np.where(upd, s, best_s)
        best_t = np.where(upd[:, None], t, best_t)
    return _finish(best_t, best_s, w.shape)


# --------------------------------------------- D: alternating refinement

def method_d(w: np.ndarray, group: int = 128, iters: int = 12,
             init: str = "c") -> tuple[np.ndarray, np.ndarray]:
    """Alternate (scale <- projection, assignment <- nearest) from a C initialisation.

    Given t, the optimal scale is <w,t>/<t,t>; given s, the optimal assignment is
    round-to-nearest. Each step is non-increasing in MSE, so this converges; it can
    still only reach a local optimum, which is why it starts from C.
    """
    g = _groups(w, group)
    if init == "c":
        t0, s0 = method_c(w, group)
        t = t0.reshape(g.shape).astype(np.float32)
        s = s0.copy()
    else:
        s = np.abs(g).mean(axis=1).astype(np.float32)
        t = _assign(g, s)

    for _ in range(iters):
        denom = (t * t).sum(axis=1)
        s_new = np.where(denom > 0, (g * t).sum(axis=1) / np.maximum(denom, 1e-30), 0.0)
        s_new = s_new.astype(np.float32)
        t_new = _assign(g, s_new)
        if np.array_equal(t_new, t) and np.allclose(s_new, s):
            s, t = s_new, t_new
            break
        s, t = s_new, t_new
    # final scale for the final support
    denom = (t * t).sum(axis=1)
    s = np.where(denom > 0, (g * t).sum(axis=1) / np.maximum(denom, 1e-30), 0.0).astype(np.float32)
    return _finish(t, s, w.shape)


METHODS = {
    "A_absmean": method_a,
    "B_mse_scale": method_b,
    "C_threshold_grid": method_c,
    "D_alternating": method_d,
}


def dequantize(t: np.ndarray, s: np.ndarray, group: int = 128) -> np.ndarray:
    shape = t.shape
    return (t.reshape(-1, group).astype(np.float32) * s[:, None]).reshape(shape)
