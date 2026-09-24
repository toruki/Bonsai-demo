"""Compact on-disk store for trained ternary expert values (experts_L{N}.npz).

The values are code * scale with code in {-1, 0, 1} and one fp16 scale per group of 128 along the
input axis, so they are kept as 2-bit codes (4 per byte) plus the fp16 scales: ~0.6 GiB per
Flash-Next layer instead of 4.8 GiB of fp16 values. load_experts() reads both this format and the
older one (plain fp16 `gate`/`up`/`down` arrays).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

GROUP = 128
NAMES = ("gate", "up", "down")


def _pack(v: np.ndarray):
    """fp16/fp32 values [..., n] -> (packed codes uint8, scales fp16 [..., n/GROUP]); exact or raises."""
    g = v.astype(np.float32).reshape(*v.shape[:-1], -1, GROUP)
    s = np.abs(g).max(-1).astype(np.float16)
    sf = s.astype(np.float32)[..., None]
    code = np.where(sf > 0, np.rint(g / np.where(sf > 0, sf, 1)), 0).astype(np.int8)
    if not np.array_equal(code.astype(np.float32) * sf, g):
        raise ValueError("values are not code * fp16 group scale")
    c = (code.reshape(-1) + 1).astype(np.uint8)                   # 0, 1, 2
    c = np.concatenate([c, np.zeros((-c.size) % 4, np.uint8)])
    packed = c[0::4] | (c[1::4] << 2) | (c[2::4] << 4) | (c[3::4] << 6)
    return packed, s


def _unpack(packed: np.ndarray, s: np.ndarray, shape) -> np.ndarray:
    n = int(np.prod(shape))
    c = np.empty(packed.size * 4, np.int8)
    for j in range(4):
        c[j::4] = (packed >> (2 * j)) & 3
    code = (c[:n] - 1).reshape(*shape[:-1], -1, GROUP)
    return (code.astype(np.float16) * s[..., None]).reshape(shape)


def save_experts(path: Path, values: dict) -> None:
    """values: {'gate','up','down'} -> arrays of ternary values. Verifies the written size."""
    arrs = {}
    for k in NAMES:
        v = values[k]
        p, s = _pack(v)
        arrs[f"{k}_codes"], arrs[f"{k}_scale"], arrs[f"{k}_shape"] = p, s, np.array(v.shape, np.int64)
    np.savez(path, **arrs)
    size, need = Path(path).stat().st_size, sum(a.nbytes for a in arrs.values())
    if size < need:                                               # a full disk truncates silently
        raise RuntimeError(f"{path} truncated: {size} < {need} bytes")


def load_experts(path: Path) -> dict:
    """-> {'gate','up','down'}: fp16 ternary values (either on-disk format)."""
    z = np.load(path)
    if "gate" in z.files:
        return {k: z[k] for k in NAMES}
    return {k: _unpack(z[f"{k}_codes"], z[f"{k}_scale"], tuple(z[f"{k}_shape"])) for k in NAMES}


if __name__ == "__main__":
    # convert old fp16 stores in place, verifying the round trip: python ternary_store.py DIR...
    import sys
    for d in sys.argv[1:]:
        for f in sorted(Path(d).glob("experts_L*.npz")):
            z = np.load(f)
            if "gate" not in z.files:
                print(f"{f}: already packed"); continue
            old = {k: z[k] for k in NAMES}
            tmp = f.with_suffix(".packed.npz")
            save_experts(tmp, old)
            new = load_experts(tmp)
            assert all(np.array_equal(old[k], new[k]) for k in NAMES), f
            before = f.stat().st_size
            tmp.replace(f)
            print(f"{f}: {before / 2**30:.2f} -> {f.stat().st_size / 2**30:.2f} GiB (round trip exact)", flush=True)
