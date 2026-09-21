"""Multi-layer block ternary-aware reconstruction on Qwen3.8-27B.

    python train_block.py --first 32 --last 35 --steps 1000

student : decoder layers first..last, all Bonsai-ternarized linears as TernaryLinear,
          fed the BF16 residual stream after layer first-1, run *sequentially* so each
          layer sees the previous ternary layer's (imperfect) output
teacher : BF16 residual stream after layer `last` (from the llama.cpp dumps; verified to
          match the PyTorch layers to ~1e-6 in train_recon.py)
loss    : rel-MSE of the block delta + lambda*(1-cos) at the block exit, plus an optional
          intermediate-exit loss (--aux) against BF16 l_out-k for k in first..last-1.

Init: PTQ (mseopt/absmean) per layer, or --init-from DIR_k to load student_state.pt from
single-layer runs (one per layer, in order).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ternary_layer import (TERNARY_LINEARS, TernaryLinear, build_layer, layer_forward, load_acts,  # noqa: E402
                           load_layer_weights, recon_metrics, shipped_layer_weights, ternarize_layer)


def block_forward(layers, x, cfg, return_all=False):
    outs = []
    h = x
    for mod in layers:
        h = layer_forward(mod, h, cfg)
        outs.append(h)
    return (h, outs) if return_all else h


@torch.no_grad()
def eval_block(layers, x, y_ref, cfg, bs=2):
    for m in layers: m.eval()
    ys = torch.cat([block_forward(layers, x[i:i + bs].cuda(), cfg).float().cpu() for i in range(0, x.shape[0], bs)])
    return {"full": recon_metrics(y_ref, ys), "delta": recon_metrics(y_ref - x, ys - x)}


def code_agreement(layers, ref_codes):
    out = {}
    for li, mod in enumerate(layers):
        for n in TERNARY_LINEARS:
            try:
                tl = mod.get_submodule(n)
            except AttributeError:
                continue
            if isinstance(tl, TernaryLinear) and (li, n) in ref_codes:
                c = tl.codes().cpu(); r = ref_codes[(li, n)]
                out[f"{li}:{n}"] = (c == r).float().mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=32)
    ap.add_argument("--last", type=int, default=35)
    ap.add_argument("--init", default="mseopt", choices=["mseopt", "absmean"])
    ap.add_argument("--init-from", nargs="*", default=None, help="student_state.pt dirs, one per layer")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr-w", type=float, default=2e-4)
    ap.add_argument("--lr-s", type=float, default=1e-4)
    ap.add_argument("--lr-norm", type=float, default=1e-4)
    ap.add_argument("--cos-weight", type=float, default=1.0)
    ap.add_argument("--aux", type=float, default=0.0, help="weight of intermediate-exit losses")
    ap.add_argument("--optim", default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--train-dir", default="/data/eval/act_train")
    ap.add_argument("--valid-dir", default="/data/eval/act_valid")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    L0, L1 = a.first, a.last
    nl = L1 - L0 + 1
    out_dir = Path(a.out or f"/data/eval/block_L{L0}-{L1}_{a.init}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = {"args": vars(a)}

    tdirs = [Path(d) for d in a.train_dir.split(",")]
    xtr = torch.cat([load_acts(d, f"l_out-{L0-1}") for d in tdirs])
    ytr = torch.cat([load_acts(d, f"l_out-{L1}") for d in tdirs])
    aux_tr = {k: torch.cat([load_acts(d, f"l_out-{k}") for d in tdirs]) for k in range(L0, L1)} if a.aux > 0 else {}
    xva, yva = load_acts(Path(a.valid_dir), f"l_out-{L0-1}"), load_acts(Path(a.valid_dir), f"l_out-{L1}")
    print(f"train {tuple(xtr.shape)} valid {tuple(xva.shape)}  layers {L0}..{L1}")

    # ---- teacher sanity + PTQ-only baseline + shipped baseline
    cfg = None
    teachers = []
    for k in range(L0, L1 + 1):
        m, cfg = build_layer(k, load_layer_weights(k)); teachers.append(m)
    tm = eval_block(teachers, xva, yva, cfg)
    print(f"teacher block vs llama.cpp: full cos {tm['full']['cos_mean']:.6f} delta relMSE {tm['delta']['rel_mse']:.2e}")
    with torch.no_grad():
        yva_t = torch.cat([block_forward(teachers, xva[i:i + 2].cuda(), cfg).float().cpu() for i in range(0, xva.shape[0], 2)])
    del teachers; torch.cuda.empty_cache()

    ship_codes = {}
    shipped = []
    for li, k in enumerate(range(L0, L1 + 1)):
        w_base = load_layer_weights(k); w_ship = shipped_layer_weights(k)
        m, _ = build_layer(k, {**w_base, **{kk: v for kk, v in w_ship.items() if not any(kk.startswith(n) for n in TERNARY_LINEARS)}})
        from ternary_layer import shipped_signs
        signs = shipped_signs()
        for n in TERNARY_LINEARS:
            if n + ".weight" in w_ship:
                parent, _, child = n.rpartition("."); wf = w_ship[n + ".weight"].cuda()
                setattr(m.get_submodule(parent), child, TernaryLinear.from_folded(wf, signs[wf.shape[1]].cuda()))
                ship_codes[(li, n)] = torch.sign(w_ship[n + ".weight"]).to(torch.int8)
        shipped.append(m)
    sm = eval_block(shipped, xva, yva_t, cfg)
    print(f"SHIPPED block: full cos {sm['full']['cos_mean']:.5f} relMSE {sm['full']['rel_mse']:.4f} | delta cos {sm['delta']['cos_mean']:.4f} relMSE {sm['delta']['rel_mse']:.4f}")
    log["shipped"] = sm
    del shipped; torch.cuda.empty_cache()

    # ---- student
    layers = []
    for li, k in enumerate(range(L0, L1 + 1)):
        m, _ = build_layer(k, load_layer_weights(k)); m = ternarize_layer(m, init=a.init)
        if a.init_from:
            sd = torch.load(Path(a.init_from[li]) / "student_state.pt")
            missing, unexpected = m.load_state_dict(sd, strict=False)
            print(f"  layer {k}: loaded {len(sd)} tensors from {a.init_from[li]} (missing {len(missing)})")
        layers.append(m)
    init_codes = {(li, n): m.get_submodule(n).codes().cpu() for li, m in enumerate(layers) for n in TERNARY_LINEARS
                  if hasattr(m, n.split('.')[0]) and isinstance(m.get_submodule(n), TernaryLinear)}
    m0 = eval_block(layers, xva, yva_t, cfg)
    print(f"INIT: full cos {m0['full']['cos_mean']:.5f} relMSE {m0['full']['rel_mse']:.4f} | delta cos {m0['delta']['cos_mean']:.4f} relMSE {m0['delta']['rel_mse']:.4f}")
    log["init"] = {"metrics": m0, "code_vs_shipped": code_agreement(layers, ship_codes)}

    p_w, p_s, p_n = [], [], []
    for m in layers:
        for name, p in m.named_parameters():
            if isinstance(m.get_submodule(name.rpartition(".")[0]), TernaryLinear):
                (p_s if name.endswith("scale") else p_w).append(p)
            elif "norm" in name:
                p_n.append(p)
            else:
                p.requires_grad_(False)
    print(f"trainable: latent {sum(p.numel() for p in p_w)/1e6:.0f}M scales {sum(p.numel() for p in p_s)/1e6:.1f}M norms {sum(p.numel() for p in p_n)}")
    groups = [{"params": p_w, "lr": a.lr_w}, {"params": p_s, "lr": a.lr_s}, {"params": p_n, "lr": a.lr_norm}]
    if a.optim == "adamw":
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.99), weight_decay=0.0)
    else:
        opt = torch.optim.SGD(groups, momentum=0.9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(t, a.steps) / a.steps)))
    print(f"GPU mem after setup: {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    n_tr = xtr.shape[0]; g = torch.Generator().manual_seed(0); hist = []; t0 = time.time()
    for m in layers: m.train()
    for step in range(1, a.steps + 1):
        idx = torch.randperm(n_tr, generator=g)[: a.bs]
        x = xtr[idx].cuda(); y_ref = ytr[idx].cuda()
        y, outs = block_forward(layers, x, cfg, return_all=True)
        y = y.float()
        d, d_ref = y - x, y_ref - x
        l_mse = ((d - d_ref) ** 2).sum() / (d_ref ** 2).sum()
        l_cos = 1 - F.cosine_similarity(y.reshape(-1, y.shape[-1]), y_ref.reshape(-1, y.shape[-1]), dim=1).mean()
        loss = l_mse + a.cos_weight * l_cos
        if a.aux > 0:
            for j, k in enumerate(range(L0, L1)):
                yk = aux_tr[k][idx].cuda(); hk = outs[j].float()
                loss = loss + a.aux * (((hk - x) - (yk - x)) ** 2).sum() / ((yk - x) ** 2).sum()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(p_w + p_s + p_n, 1.0)
        opt.step(); sched.step()
        if step % 20 == 0:
            print(f"step {step:5d} loss {loss.item():.4f} delta_relmse {l_mse.item():.4f} 1-cos {l_cos.item():.5f} ({time.time()-t0:.0f}s) mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            m = eval_block(layers, xva, yva_t, cfg)
            for mm in layers: mm.train()
            ca = code_agreement(layers, ship_codes); ci = code_agreement(layers, init_codes)
            hist.append({"step": step, "metrics": m, "code_vs_shipped": ca, "code_vs_init": ci})
            print(f"  [eval {step}] full cos {m['full']['cos_mean']:.5f} relMSE {m['full']['rel_mse']:.4f} | delta cos {m['delta']['cos_mean']:.4f} relMSE {m['delta']['rel_mse']:.4f} | "
                  f"code agree shipped {sum(ca.values())/len(ca):.4f} init {sum(ci.values())/len(ci):.4f}", flush=True)
    log["history"] = hist
    for li, m in enumerate(layers):
        torch.save({k: v.detach().cpu() for k, v in m.state_dict().items() if not k.endswith((".H", ".signs"))}, out_dir / f"student_state_L{L0+li}.pt")
    (out_dir / "log.json").write_text(json.dumps(log, indent=1))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
