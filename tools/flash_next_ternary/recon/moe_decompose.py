"""Split the routed-MoE output error of a downstream (unmodified) layer into a routing part and an
input-drift part.

For layer L the teacher layer (IQ3_XXS, dequantised) is run on the variant's own stream entering L
(l_last-{L-1} from the variant dump) up to the MoE input h_var. Then

  y_nat  = MoE(h_var, variant routing: own top-k + weights)      ~ the variant's ffn_moe_out-L
  y_forc = MoE(h_var, reference routing: reference top-k + weights, from the reference dump)

  total   = |y_var_dump - y_ref_dump|^2 / |y_ref_dump|^2
  routing = |y_nat - y_forc|^2          / |y_ref_dump|^2   (only which experts / what gate weights)
  drift   = |y_forc - y_ref_dump|^2     / |y_ref_dump|^2   (same experts, same weights, shifted input)

Two checks come with it: `repro` = |y_nat - y_var_dump|^2 / |y_var_dump|^2 (python vs llama.cpp on the
variant, includes llama.cpp's q8 activation quantisation), and `floor` = the same for the reference
model on its own stream.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from q4x_layer import FLASH_GGUF, GGUFTensors, LayerRunner, build_layer, layer_weights, text_config


def load(d: Path, name: str, dtype, width: int):
    return np.fromfile(d / f"{name}.{'i32' if dtype == np.int32 else 'f32'}", dtype).reshape(-1, width)


@torch.no_grad()
def moe_input(mod, runner, x):
    """stream [B, T, hc*H] -> MoE input [B*T, H] and router logits (teacher layer, attention half)."""
    h, hin, inj = mod.attn_hyper_connection(x)
    if mod.layer_type == "linear_attention":
        h = mod.linear_attn(h, cache_params=None, attention_mask=None)
    else:
        pe, mask = runner.pe_mask(x)
        h, _ = mod.self_attn(h, pe, attention_mask=mask)
    x2 = hin + (h.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
    h2, _, _ = mod.mlp_hyper_connection(x2)
    h2 = h2.reshape(-1, h2.shape[-1])
    return h2, F.linear(h2, mod.mlp.gate.weight)


def rel(a, b, ref):
    return float(((a - b) ** 2).sum() / (ref ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("var")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    R, V = Path(a.ref), Path(a.var)
    cfg = text_config(); H, W, k = cfg.hidden_size, cfg.hidden_size * cfg.hc_count, cfg.num_experts_per_tok
    dev = "cuda"
    runner = LayerRunner(cfg, dev)
    g = GGUFTensors(FLASH_GGUF)
    out = {}
    print(f"{'layer':6s} {'total':>8s} {'routing':>8s} {'drift':>8s} {'rout/tot':>8s} {'repro':>8s} {'floor':>8s} {'topk agree':>10s}")
    for L in a.layers:
        mod, _, miss, unexp = build_layer(L, layer_weights(g, L, cfg), device=dev)
        assert not miss and not unexp
        res = {}
        for side, D in (("var", V), ("ref", R)):
            x = torch.from_numpy(load(D, f"l_last-{L - 1}", np.float32, W)).reshape(-1, a.seq_len, W)
            hs, lg = [], []
            for i in range(0, x.shape[0], 2):
                h, l = moe_input(mod, runner, x[i:i + 2].to(dev))
                hs.append(h); lg.append(l)
            res[side] = (torch.cat(hs), torch.cat(lg))
        yv = torch.from_numpy(load(V, f"ffn_moe_out-{L}", np.float32, H)).to(dev)
        yr = torch.from_numpy(load(R, f"ffn_moe_out-{L}", np.float32, H)).to(dev)
        tv = torch.from_numpy(load(V, f"ffn_moe_topk-{L}", np.int32, k)).to(dev).long()
        wv = torch.from_numpy(load(V, f"ffn_moe_weights_norm-{L}", np.float32, k)).to(dev)
        tr = torch.from_numpy(load(R, f"ffn_moe_topk-{L}", np.int32, k)).to(dev).long()
        wr = torch.from_numpy(load(R, f"ffn_moe_weights_norm-{L}", np.float32, k)).to(dev)
        ex = mod.mlp.experts
        with torch.no_grad():
            hv, lv = res["var"]; hr, lr = res["ref"]
            # does our router reproduce the dumped top-k? (sets, per token)
            own = lv.softmax(-1).topk(k, -1).indices
            agree = float((own.sort(-1).values == tv.sort(-1).values).all(-1).float().mean())
            y_nat = ex(hv, tv, wv)            # dumped variant routing (= what llama.cpp used)
            y_forc = ex(hv, tr, wr)           # reference routing forced onto the variant's input
            y_floor = ex(hr, tr, wr)
        d = {"total": rel(yv, yr, yr), "routing": rel(y_nat, y_forc, yr), "drift": rel(y_forc, yr, yr),
             "repro": rel(y_nat, yv, yv), "floor": rel(y_floor, yr, yr), "topk_agree": agree}
        out[L] = d
        print(f"L{L:<5d} {d['total']:8.4f} {d['routing']:8.4f} {d['drift']:8.4f} {d['routing']/d['total']:8.2f} "
              f"{d['repro']:8.4f} {d['floor']:8.4f} {agree:10.3f}", flush=True)
        del mod, ex, res; torch.cuda.empty_cache()
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
