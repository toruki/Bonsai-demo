"""Phase 3: single-tensor ternary experiment on a Qwen3.8-Flash-Next routed expert.

Pipeline under test (all in float32, no model is modified):

    BF16 W  ->  sign flip  ->  blockwise Hadamard  ->  group-128 ternary  ->  dequantize

The rotation is the one the PrismML runtime actually performs
(`bonsai_format.fold_weight` / `rotate_activation`, transcribed from llama-model.cpp
and llama-graph.cpp), not a scipy look-alike. Correctness of the pair is asserted
before anything is measured.

Activations here are SYNTHETIC. Capturing real layer-0 expert inputs needs a running
qwen4exp graph, which the PrismML fork does not have yet (docs/flash_next_quantization_plan.md
§0-1); that is a Phase 4 item. Two synthetic regimes are used:
  iid      -- N(0,1) per feature
  outlier  -- per-channel gains ~ lognormal, which is what makes rotation worth doing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bonsai_format import (fold_weight, hadamard_matrix, ptq1_0_quantize,  # noqa: E402
                           pq2_0_quantize, rotate_activation)
from fetch_slice import REPO, fetch  # noqa: E402
from ternary_quant import METHODS, dequantize  # noqa: E402


# --------------------------------------------------------------------- metrics

def weight_metrics(w: np.ndarray, wq: np.ndarray, t: np.ndarray, s: np.ndarray) -> dict:
    err = wq - w
    mse = float((err ** 2).mean())
    denom = float((w ** 2).mean())
    a, b = w.reshape(-1), wq.reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    nz = s[s > 0]
    return {
        "mse": mse,
        "rel_mse": mse / denom,
        "cosine": cos,
        "max_abs_err": float(np.abs(err).max()),
        "zero_frac": float((t == 0).mean()),
        "pos_frac": float((t == 1).mean()),
        "neg_frac": float((t == -1).mean()),
        "scale_min": float(nz.min()) if nz.size else 0.0,
        "scale_med": float(np.median(nz)) if nz.size else 0.0,
        "scale_max": float(nz.max()) if nz.size else 0.0,
        "scale_p99_over_p1": (float(np.percentile(nz, 99) / max(np.percentile(nz, 1), 1e-30))
                              if nz.size else 0.0),
    }


def output_metrics(y: np.ndarray, yq: np.ndarray) -> dict:
    err = yq - y
    mse = float((err ** 2).mean())
    a, b = y.reshape(-1), yq.reshape(-1)
    return {
        "out_mse": mse,
        "out_rel_mse": mse / float((y ** 2).mean()),
        "out_rel_fro": float(np.linalg.norm(err) / np.linalg.norm(y)),
        "out_cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)),
        "out_max_abs_err": float(np.abs(err).max()),
    }


# ------------------------------------------------------------------ activations

def make_activations(n_tok: int, n_in: int, regime: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_tok, n_in)).astype(np.float32)
    if regime == "outlier":
        gains = np.exp(rng.standard_normal(n_in).astype(np.float32) * 1.2)
        idx = rng.choice(n_in, size=max(1, n_in // 256), replace=False)
        gains[idx] *= 20.0                      # a few massive channels, as in real LLMs
        x = x * gains
    # RMSNorm-like normalisation, which is what actually precedes these projections
    x = x / (np.sqrt((x ** 2).mean(axis=1, keepdims=True)) + 1e-6)
    return x


# ------------------------------------------------------------------------ run

def run_one(w: np.ndarray, block: int | None, group: int, signs: np.ndarray | None,
            acts: dict[str, np.ndarray]) -> dict:
    """block=None -> no rotation (control)."""
    wf = w if block is None else fold_weight(w, block, signs)
    res = {}
    for name, fn in METHODS.items():
        t, s = fn(wf, group)
        wq = dequantize(t, s, group)
        m = weight_metrics(wf, wq, t, s)

        for reg, x in acts.items():
            xr = x if block is None else rotate_activation(x, block, signs)
            y = x @ w.T                                     # exact reference
            yq = xr @ wq.T
            for k, v in output_metrics(y, yq).items():
                m[f"{k}[{reg}]"] = v
        res[name] = m
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--part", choices=["gate", "up", "down"], default="gate")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--blocks", type=int, nargs="*", default=None,
                    help="Hadamard block sizes to try; 0 means 'no rotation'")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    # ---- pull the weight
    if a.part == "down":
        name = f"model.language_model.layers.{a.layer}.mlp.experts.down_proj"
        w = fetch(REPO, name, a.expert, 1)[0]               # [2560, 640]
    else:
        name = f"model.language_model.layers.{a.layer}.mlp.experts.gate_up_proj"
        gu = fetch(REPO, name, a.expert, 1)[0]              # [1280, 2560]
        half = gu.shape[0] // 2
        w = gu[:half] if a.part == "gate" else gu[half:]
    w = np.ascontiguousarray(w, dtype=np.float32)
    n_out, n_in = w.shape

    blocks = a.blocks
    if blocks is None:
        blocks = [0] + [b for b in (128, 256, 512, 1024, 2048) if n_in % b == 0]

    print(f"tensor : {name} [expert {a.expert}, {a.part}]")
    print(f"shape  : {w.shape}  (n_in={n_in}, n_out={n_out})")
    print(f"stats  : std={w.std():.5g} absmax={np.abs(w).max():.5g} "
          f"kurtosis={float(((w-w.mean())**4).mean()/w.var()**2):.3f}")
    print(f"group  : {a.group}   blocks tried: {blocks}")
    print()

    rng = np.random.default_rng(a.seed)
    signs = rng.choice(np.array([-1.0, 1.0], np.float32), n_in)
    acts = {reg: make_activations(a.tokens, n_in, reg, a.seed + 7)
            for reg in ("iid", "outlier")}

    # sanity: the rotation/fold pair must be an exact identity before we trust anything
    for b in [x for x in blocks if x]:
        lhs = rotate_activation(acts["iid"][:8], b, signs) @ fold_weight(w, b, signs).T
        rhs = acts["iid"][:8] @ w.T
        rel = np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs)
        assert rel < 2e-5, f"fold identity broken at block {b}: rel={rel:.3g}"
    print("fold/rotate identity verified for every block size\n")

    all_res = {}
    hdr = (f"{'block':>6s} {'method':<17s} {'rel_mse':>9s} {'cosine':>9s} {'zero%':>7s} "
           f"{'max_err':>9s} {'s_p99/p1':>9s} {'out_rel[iid]':>13s} {'out_rel[out]':>13s} "
           f"{'out_cos[out]':>13s}")
    print(hdr)
    print("-" * len(hdr))
    for b in blocks:
        res = run_one(w, b or None, a.group, signs if b else None, acts)
        all_res[str(b)] = res
        for m, v in res.items():
            print(f"{(b or '-'):>6} {m:<17s} {v['rel_mse']:9.5f} {v['cosine']:9.6f} "
                  f"{100*v['zero_frac']:7.2f} {v['max_abs_err']:9.5f} "
                  f"{v['scale_p99_over_p1']:9.3f} {v['out_rel_mse[iid]']:13.5f} "
                  f"{v['out_rel_mse[outlier]']:13.5f} {v['out_cosine[outlier]']:13.6f}")
        print()

    # ---- packing check on the best config: the ternary result must survive PQ2_0/PTQ1_0
    best_b = max((b for b in blocks if b), default=0)
    if best_b:
        t, s = METHODS["D_alternating"](fold_weight(w, best_b, signs), a.group)
        wq = dequantize(t, s, a.group)
        from bonsai_format import pq2_0_dequantize, ptq1_0_dequantize
        ok_pq = np.array_equal(pq2_0_dequantize(pq2_0_quantize(wq)), wq.reshape(-1))
        ok_pt = np.array_equal(ptq1_0_dequantize(ptq1_0_quantize(wq)), wq.reshape(-1))
        print(f"packing round-trip @ block {best_b}: PQ2_0 exact={ok_pq}  PTQ1_0 exact={ok_pt}")
        bpw_pq = 34 * 8 / a.group
        bpw_pt = 28 * 8 / a.group
        print(f"storage: PQ2_0 {bpw_pq:.3f} bpw, PTQ1_0 {bpw_pt:.3f} bpw "
              f"(BF16 reference 16.0)")

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(all_res, indent=1))
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
