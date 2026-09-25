"""Compact on-disk store for trained expert values (experts_L{N}.npz).

Values are code * scale with a fp16 scale per group of GROUP along the input axis and codes in
{-1, 0, 1} (ternary) or {-1, 0, 1, 2} (the PQ2_0 levels). They are kept as 2-bit codes (4 per byte,
stored as code + 1) plus the fp16 scales: ~0.6 GiB per Flash-Next layer.

v2 files (`codes` + `scale` per tensor) store the scales explicitly, so a code of +2 survives; the older
v1 files derived the scale from the values (amax), which only works for ternary. load_experts() reads
both and returns the values plus, for v2, the codes and scales.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

GROUP = 128
NAMES = ("gate", "up", "down")


def _pack_codes(code: np.ndarray) -> np.ndarray:
    """int8 codes in [-1, 2] -> uint8, 4 per byte (stored as code + 1)."""
    c = (code.reshape(-1).astype(np.int16) + 1)
    if c.min() < 0 or c.max() > 3:
        raise ValueError("codes must lie in [-1, 2]")
    c = c.astype(np.uint8)
    c = np.concatenate([c, np.zeros((-c.size) % 4, np.uint8)])
    return c[0::4] | (c[1::4] << 2) | (c[2::4] << 4) | (c[3::4] << 6)


def _unpack_codes(packed: np.ndarray, shape) -> np.ndarray:
    n = int(np.prod(shape))
    c = np.empty(packed.size * 4, np.int8)
    for j in range(4):
        c[j::4] = (packed >> (2 * j)) & 3
    return (c[:n] - 1).reshape(shape)


def values_from(codes: dict, scales: dict) -> dict:
    """{'gate','up','down'} codes [..., n] + fp16 scales [..., n/GROUP] -> fp16 values."""
    out = {}
    for k in NAMES:
        c, s = codes[k], np.asarray(scales[k], np.float16)
        out[k] = (c.reshape(*c.shape[:-1], -1, GROUP).astype(np.float16) * s[..., None]).reshape(c.shape)
    return out


def save_experts(path, codes: dict, scales: dict) -> None:
    """v2: explicit codes + fp16 scales per tensor. Verifies the written size."""
    arrs = {"version": np.array(2)}
    for k in NAMES:
        c = np.asarray(codes[k], np.int8)
        arrs[f"{k}_codes"], arrs[f"{k}_scale"] = _pack_codes(c), np.asarray(scales[k], np.float16)
        arrs[f"{k}_shape"] = np.array(c.shape, np.int64)
    path = Path(path)
    np.savez(path, **arrs)
    need = sum(a.nbytes for a in arrs.values())
    if path.stat().st_size < need:                                 # a full disk truncates silently
        raise RuntimeError(f"{path} truncated: {path.stat().st_size} < {need} bytes")


def load_experts(path) -> dict:
    """-> {'gate','up','down'}: fp16 values; v2 files also carry '<k>_codes' (int8) and '<k>_scale' (fp16)."""
    z = np.load(path)
    if "gate" in z.files:                                          # v0: plain fp16 values
        return {k: z[k] for k in NAMES}
    out = {}
    v2 = "version" in z.files
    for k in NAMES:
        shape = tuple(z[f"{k}_shape"])
        c, s = _unpack_codes(z[f"{k}_codes"], shape), z[f"{k}_scale"]
        out[k] = (c.reshape(*shape[:-1], -1, GROUP).astype(np.float16) * s[..., None]).reshape(shape)
        if v2:
            out[f"{k}_codes"], out[f"{k}_scale"] = c, s
    return out


if __name__ == "__main__":
    # self-test of the v2 round trip, including +2 codes
    rng = np.random.default_rng(0)
    codes = {"gate": rng.integers(-1, 2, (3, 8, 256), dtype=np.int8), "up": rng.integers(-1, 2, (3, 8, 256), dtype=np.int8),
             "down": rng.integers(-1, 3, (3, 256, 128), dtype=np.int8)}
    scales = {k: rng.random(c.shape[:-1] + (c.shape[-1] // GROUP,)).astype(np.float16) for k, c in codes.items()}
    save_experts("/tmp/ts_selftest.npz", codes, scales)
    z = load_experts("/tmp/ts_selftest.npz"); v = values_from(codes, scales)
    ok = all(np.array_equal(z[k], v[k]) and np.array_equal(z[f"{k}_codes"], codes[k]) and np.array_equal(z[f"{k}_scale"], scales[k])
             for k in NAMES)
    print("v2 round trip exact:", ok, "| down has +2:", bool((z["down_codes"] == 2).any()))
