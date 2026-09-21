"""Reverse-analyse Bonsai 2 27B against its base model Qwen3.8-27B.

Bonsai 2 ships ternary codes t and fp16 group scales d in a rotated basis (Phase 1).
The base checkpoint Qwen/Qwen3.8-27B is public. If Bonsai 2 were a post-training
quantization of that checkpoint, then for the correct fold convention

    W' = fold(W_base)   ->   t = Q(W' / d)

must hold with a clean decision boundary in W'/d. This script

  1. decodes the shipped codes for a row range of one tensor,
  2. folds the same rows of the base weight under several candidate conventions
     (with/without the shipped sign vector, sign-then-H vs H-then-sign, with the
     preceding RMSNorm weight folded in or not),
  3. reports, per convention, how well W' explains t (correlation, sign agreement,
     zero/non-zero separability), and
  4. for the best convention, characterises the boundary: how many elements would
     a single per-group threshold misclassify, and where it sits relative to d.

Everything is measured on real shipped bytes; nothing is fitted to the answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "llama.cpp" / "gguf-py"))
from bonsai_format import PQ2_0_BLOCK_BYTES, QK_PQ2_0, hadamard_matrix, pq2_0_codes  # noqa: E402
from fetch_slice import fetch  # noqa: E402
from gguf import GGUFReader  # noqa: E402

BASE_REPO = "Qwen/Qwen3.8-27B"
GGUF = "models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"

# GGUF name -> (HF name, HF name of the RMSNorm feeding it)
MAP = {
    "blk.{l}.ffn_gate.weight": ("model.language_model.layers.{l}.mlp.gate_proj.weight",
                                "model.language_model.layers.{l}.post_attention_layernorm.weight"),
    "blk.{l}.ffn_up.weight":   ("model.language_model.layers.{l}.mlp.up_proj.weight",
                                "model.language_model.layers.{l}.post_attention_layernorm.weight"),
    "blk.{l}.ffn_down.weight": ("model.language_model.layers.{l}.mlp.down_proj.weight", None),
}


def load_bonsai(reader: GGUFReader, name: str, rows: int):
    t = [x for x in reader.tensors if x.name == name][0]
    ne0 = int(t.shape[0])
    nblk = ne0 // QK_PQ2_0
    raw = t.data.view(np.uint8).reshape(-1, PQ2_0_BLOCK_BYTES)[: rows * nblk]
    codes, d = pq2_0_codes(raw)
    return codes.reshape(rows, ne0), d.reshape(rows, nblk), ne0


def signs_for(reader: GGUFReader, width: int) -> np.ndarray:
    widths = reader.fields["prism.hadamard.sign_widths"].contents()
    vals = np.asarray(reader.fields["prism.hadamard.sign_values"].contents(), dtype=np.float32)
    off = 0
    for w in widths:
        if w == width:
            return vals[off:off + w]
        off += w
    raise KeyError(width)


def fold_variants(w: np.ndarray, block: int, s: np.ndarray, norm: np.ndarray | None):
    h = hadamard_matrix(block)

    def blk(m):
        return (m.reshape(-1, block) @ h).reshape(m.shape)

    v = {
        "H only (no sign)":       blk(w),
        "sign then H  [runtime]": blk(w * s),
        "H then sign":            blk(w) * s,
        "no rotation":            w,
    }
    if norm is not None:
        v["norm*W, sign then H"] = blk((w * norm) * s)
    return v


def explain(codes: np.ndarray, d: np.ndarray, wp: np.ndarray) -> dict:
    """How well does wp (same shape as codes) explain the ternary codes?"""
    g = QK_PQ2_0
    rows, n = codes.shape
    u = (wp.reshape(rows, n // g, g) / np.maximum(d, 1e-12)[:, :, None]).reshape(rows, n)
    deq = (codes.reshape(rows, n // g, g) * d[:, :, None]).reshape(rows, n)
    a, b = wp.ravel(), deq.ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    nz = codes != 0
    sign_agree = float((np.sign(u[nz]) == codes[nz]).mean())
    # separability of |u| between t==0 and t!=0: AUC via rank statistic
    au = np.abs(u).ravel()
    lab = nz.ravel()
    order = np.argsort(au, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, au.size + 1)
    n1, n0 = lab.sum(), (~lab).sum()
    auc = float((ranks[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return {"corr": corr, "sign_agree": sign_agree, "auc_zero_vs_nonzero": auc}


def boundary(codes: np.ndarray, d: np.ndarray, wp: np.ndarray) -> dict:
    """Per-group: best single threshold on |W'|/d and its misclassification rate."""
    g = QK_PQ2_0
    rows, n = codes.shape
    u = (wp.reshape(-1, g) / np.maximum(d.reshape(-1, 1), 1e-12))
    c = codes.reshape(-1, g)
    nz = c != 0
    au = np.abs(u)
    # best threshold per group: sort |u|, the ideal cut puts all zeros below
    best_err = np.zeros(u.shape[0])
    best_th = np.zeros(u.shape[0])
    for i in range(u.shape[0]):
        o = np.argsort(au[i])
        lab = nz[i][o].astype(np.int64)          # 1 = nonzero
        # errors if cut after position k: zeros above k (lab==0 in tail) + nonzeros below
        nonzero_below = np.cumsum(lab)
        zero_above = (lab == 0).sum() - np.cumsum(lab == 0)
        err = np.concatenate([[lab.sum()], nonzero_below + zero_above])  # cut before 0..after all
        k = int(err.argmin())
        best_err[i] = err[k]
        best_th[i] = au[i][o][k - 1] if k > 0 else 0.0
    zero_bounds = au[nz == 0]
    nz_bounds = au[nz]
    return {
        "single_threshold_misclass_rate": float(best_err.sum() / c.size),
        "groups_perfectly_separable": float((best_err == 0).mean()),
        "threshold_over_d_median": float(np.median(best_th)),
        "threshold_over_d_p10_p90": [float(np.percentile(best_th, 10)), float(np.percentile(best_th, 90))],
        "|u| of zeros   p50/p90/p99": [float(np.percentile(zero_bounds, p)) for p in (50, 90, 99)],
        "|u| of nonzero p1/p10/p50":  [float(np.percentile(nz_bounds, p)) for p in (1, 10, 50)],
        "sign flips (t = -sign(W'))": float((np.sign(u[nz]) != c[nz]).mean()),
        "d / rms(W') median":         float(np.median(d.reshape(-1) / np.sqrt((wp.reshape(-1, g) ** 2).mean(axis=1)))),
        "d / mean|W'| median":        float(np.median(d.reshape(-1) / np.abs(wp.reshape(-1, g)).mean(axis=1))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tensor", default="blk.{l}.ffn_gate.weight")
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    r = GGUFReader(GGUF)
    block = int(r.fields["prism.hadamard.block_size"].contents())
    gname = a.tensor.format(l=a.layer)
    hf_name, norm_name = MAP[a.tensor]
    hf_name = hf_name.format(l=a.layer)

    codes, d, ne0 = load_bonsai(r, gname, a.rows)
    s = signs_for(r, ne0)
    w = fetch(BASE_REPO, hf_name, 0, a.rows).astype(np.float32)
    assert w.shape == codes.shape, (w.shape, codes.shape)
    norm = None
    if norm_name:
        norm = fetch(BASE_REPO, norm_name.format(l=a.layer)).astype(np.float32) + 1.0  # Qwen3.5 stores w-1

    print(f"bonsai tensor : {gname}  rows 0..{a.rows-1}  (ne0={ne0}, block={block})")
    print(f"base tensor   : {hf_name}")
    print(f"zero rate in these rows: {(codes == 0).mean():.4f}\n")

    out = {}
    print(f"{'fold convention':26s} {'corr(Wp,t*d)':>13s} {'sign agree':>11s} {'AUC 0/nz':>9s}")
    for name, wp in fold_variants(w, block, s, norm).items():
        m = explain(codes, d, wp)
        out[name] = m
        print(f"{name:26s} {m['corr']:13.5f} {m['sign_agree']:11.5f} {m['auc_zero_vs_nonzero']:9.5f}")

    best = max(out, key=lambda k: out[k]["corr"])
    wp = fold_variants(w, block, s, norm)[best]
    print(f"\nbest convention: {best}\n")
    b = boundary(codes, d, wp)
    out["boundary[" + best + "]"] = b
    for k, v in b.items():
        print(f"  {k:36s} {v}")

    # the base weight's own group statistics vs Bonsai's scale, to test scale rules
    g = QK_PQ2_0
    wg = wp.reshape(-1, g)
    for label, stat in [("amax", np.abs(wg).max(axis=1)),
                        ("mean|w|", np.abs(wg).mean(axis=1)),
                        ("rms", np.sqrt((wg ** 2).mean(axis=1)))]:
        ratio = d.reshape(-1) / np.maximum(stat, 1e-12)
        print(f"  d / {label:8s}: median {np.median(ratio):.4f}  IQR [{np.percentile(ratio,25):.4f}, "
              f"{np.percentile(ratio,75):.4f}]  CV {ratio.std()/ratio.mean():.3f}")

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
