"""Gate-weighted routing comparison of a variant against the reference model, per layer.

Reads dump_hidden dumps (same tokens for both) of
  ffn_moe_probs-N         [n_tok, 512]  full router softmax
  ffn_moe_topk-N          [n_tok, k]    selected experts (i32)
  ffn_moe_weights_norm-N  [n_tok, k]    gate weights after top-k renormalisation
  ffn_moe_out-N           [n_tok, H]    routed-expert output (sum_i gate_i * expert_i(x), no shared expert)
  l_last-N                [n_tok, hc*H] residual stream after layer N (for the input drift of layer N+1)

and reports per layer
  set_change     1 - |top-k_ref ∩ top-k_var| / k                      (the old metric)
  changed_mass   gate mass on experts that left or entered the top-k, averaged over both sides
  sparse_tv      0.5 * sum_e |w_ref(e) - w_var(e)| over the sparse 512-expert gate vectors
                 (membership change + reweighting of the kept experts)
  full_tv / full_kl   distance of the full 512-way router softmax
  moe_rel / moe_cos   relative MSE and mean per-token cosine of the routed-expert output
  in_rel         relative MSE of the stream entering the layer (upstream drift)
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(d: Path, name: str, dtype, width: int):
    f = d / f"{name}.{'i32' if dtype == np.int32 else 'f32'}"
    if not f.exists():
        return None
    x = np.fromfile(f, dtype)
    return x.reshape(-1, width) if x.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("var")
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=2560); ap.add_argument("--hc", type=int, default=4)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--first", type=int, default=4, help="first replaced layer (for the summary groups)")
    ap.add_argument("--last", type=int, default=11, help="last replaced layer")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    R, V = Path(a.ref), Path(a.var)
    res = {}
    for L in range(a.layers):
        tr, tv = load(R, f"ffn_moe_topk-{L}", np.int32, a.k), load(V, f"ffn_moe_topk-{L}", np.int32, a.k)
        if tr is None or tv is None:
            continue
        wr, wv = load(R, f"ffn_moe_weights_norm-{L}", np.float32, a.k), load(V, f"ffn_moe_weights_norm-{L}", np.float32, a.k)
        n = tr.shape[0]
        # sparse gate vectors over all experts
        gr = np.zeros((n, a.experts), np.float32); gv = np.zeros((n, a.experts), np.float32)
        rows = np.arange(n)[:, None]
        gr[rows, tr] = wr; gv[rows, tv] = wv
        inr, inv = gr > 0, gv > 0
        # rows with an exactly-zero kept weight are vanishingly rare; membership from the index sets instead
        mr = np.zeros_like(inr); mr[rows, tr] = True
        mv = np.zeros_like(inv); mv[rows, tv] = True
        left, entered = mr & ~mv, mv & ~mr
        set_change = 1 - (mr & mv).sum(1) / a.k
        changed = 0.5 * ((gr * left).sum(1) + (gv * entered).sum(1))
        stv = 0.5 * np.abs(gr - gv).sum(1)
        d = {"set_change": float(set_change.mean()), "changed_mass": float(changed.mean()),
             "changed_mass_p90": float(np.quantile(changed, 0.9)),
             "tok_mass_gt_0.1": float((changed > 0.1).mean()), "sparse_tv": float(stv.mean())}
        pr, pv = load(R, f"ffn_moe_probs-{L}", np.float32, a.experts), load(V, f"ffn_moe_probs-{L}", np.float32, a.experts)
        if pr is not None and pv is not None:
            d["full_tv"] = float(0.5 * np.abs(pr - pv).sum(1).mean())
            d["full_kl"] = float((pr * (np.log(np.maximum(pr, 1e-30)) - np.log(np.maximum(pv, 1e-30)))).sum(1).mean())
        orr, ovv = load(R, f"ffn_moe_out-{L}", np.float32, a.hidden), load(V, f"ffn_moe_out-{L}", np.float32, a.hidden)
        if orr is not None and ovv is not None:
            d["moe_rel"] = float(((ovv - orr) ** 2).sum() / (orr ** 2).sum())
            num = (orr * ovv).sum(1); den = np.linalg.norm(orr, axis=1) * np.linalg.norm(ovv, axis=1)
            d["moe_cos"] = float((num / np.maximum(den, 1e-30)).mean())
        if L > 0:
            sr = load(R, f"l_last-{L - 1}", np.float32, a.hidden * a.hc)
            sv = load(V, f"l_last-{L - 1}", np.float32, a.hidden * a.hc)
            if sr is not None and sv is not None:
                d["in_rel"] = float(((sv - sr) ** 2).sum() / (sr ** 2).sum())
        res[L] = d

    keys = ["set_change", "changed_mass", "changed_mass_p90", "tok_mass_gt_0.1", "sparse_tv", "full_tv", "full_kl",
            "in_rel", "moe_rel", "moe_cos"]
    print("layer " + " ".join(f"{k:>12s}" for k in keys))
    for L, d in res.items():
        print(f"L{L:<4d} " + " ".join(f"{d.get(k, float('nan')):12.4f}" for k in keys))
    f, l, n = a.first, a.last, a.layers - 1
    for name, lo, hi in ((f"0-{f-1}", 0, f - 1), (f"{f}", f, f), (f"{f+1}-{l}", f + 1, l), (f"{l+1}-{n}", l + 1, n)):
        sel = [L for L in res if lo <= L <= hi]
        if sel:
            print(f"mean {name:6s}" + " ".join(f"{np.mean([res[L].get(k, np.nan) for L in sel]):12.4f}" for k in keys))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
