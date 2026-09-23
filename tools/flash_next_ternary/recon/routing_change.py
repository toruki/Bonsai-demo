"""Per-layer routing change of a variant vs the IQ3_XXS reference, from ffn_moe_topk-N.i32 dumps.
Reports mean |top-k set difference| / k (0 = identical routing) and the fraction of tokens whose
top-1 expert changed."""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("ref"); ap.add_argument("var"); ap.add_argument("--k", type=int, default=10)
ap.add_argument("--json", default=None)
a = ap.parse_args()
res = {}
for L in range(48):
    fr, fv = Path(a.ref) / f"ffn_moe_topk-{L}.i32", Path(a.var) / f"ffn_moe_topk-{L}.i32"
    if not fr.exists() or not fv.exists():
        continue
    r = np.fromfile(fr, np.int32).reshape(-1, a.k); v = np.fromfile(fv, np.int32).reshape(-1, a.k)
    assert r.shape == v.shape, (L, r.shape, v.shape)
    if r.size == 0:                 # last layer is only evaluated on output rows (none with --no-logits)
        continue
    rs, vs = np.sort(r, 1), np.sort(v, 1)
    overlap = np.array([np.intersect1d(x, y, assume_unique=True).size for x, y in zip(rs, vs)])
    res[L] = {"set_change": float(1 - overlap.mean() / a.k), "top1_change": float((r[:, 0] != v[:, 0]).mean())}
for L, d in res.items():
    print(f"L{L:2d}  set change {d['set_change']:.4f}  top1 change {d['top1_change']:.4f}")
for name, sel in (("all", list(res)), ("0-3", [L for L in res if L < 4]), ("4-11", [L for L in res if 4 <= L <= 11]),
                  ("12+", [L for L in res if L >= 12])):
    if sel:
        print(f"mean over {name:4s} ({len(sel):2d} layers): set change {np.mean([res[L]['set_change'] for L in sel]):.4f}"
              f"  top1 change {np.mean([res[L]['top1_change'] for L in sel]):.4f}")
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
