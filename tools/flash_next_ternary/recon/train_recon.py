"""Single-layer ternary-aware output reconstruction on Qwen3.8-27B.

    python train_recon.py --layer 32 --init mseopt --steps 600

teacher : BF16 decoder layer L (fp32 compute), input x = BF16 residual stream after L-1
student : same layer with TernaryLinear (latent weights + group scales + RMSNorm trained)
loss    : rel-MSE of the block delta f(x) = y - x  +  lambda * (1 - cos(y, y_ref))

Reports before/after: block-output cos / rel-MSE (full and delta), and ternary code
agreement with (a) the PTQ initialisation and (b) the shipped Bonsai 2 layer. Also
evaluates the shipped Bonsai 2 layer itself on the same inputs as the reference bar.
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
from ternary_layer import (GGUF_NAME, TERNARY_LINEARS, TernaryLinear, build_layer, layer_forward,  # noqa: E402
                           load_acts, load_layer_weights, recon_metrics, shipped_layer_weights,
                           shipped_signs, ternarize_layer)


def all_metrics(mod, x, y_ref, cfg, bs=4) -> dict:
    mod.eval()
    ys = []
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            ys.append(layer_forward(mod, x[i:i + bs].cuda(), cfg).float().cpu())
    y = torch.cat(ys)
    full = recon_metrics(y_ref, y)
    delta = recon_metrics(y_ref - x, y - x)
    return {"full": full, "delta": delta}


def code_agreement(mod, ref_codes: dict[str, torch.Tensor]) -> dict[str, float]:
    out = {}
    for name in TERNARY_LINEARS:
        try:
            tl = mod.get_submodule(name)
        except AttributeError:
            continue
        if not isinstance(tl, TernaryLinear) or name not in ref_codes:
            continue
        c = tl.codes().cpu()
        r = ref_codes[name]
        out[name] = {"agree": (c == r).float().mean().item(), "zero": (c == 0).float().mean().item(),
                     "sign_flip": ((c * r) == -1).float().mean().item()}
    return out


def student_codes(mod) -> dict[str, torch.Tensor]:
    return {n: mod.get_submodule(n).codes().cpu() for n in TERNARY_LINEARS
            if hasattr(mod, n.split(".")[0]) and isinstance(mod.get_submodule(n), TernaryLinear)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--init", default="mseopt", choices=["mseopt", "absmean", "shipped"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr-w", type=float, default=2e-4)
    ap.add_argument("--lr-s", type=float, default=1e-4)
    ap.add_argument("--lr-norm", type=float, default=1e-4)
    ap.add_argument("--cos-weight", type=float, default=1.0)
    ap.add_argument("--freeze-codes", action="store_true", help="train scales+norms only (control)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--train-dir", default="/data/eval/act_train", help="comma-separated dump dirs")
    ap.add_argument("--valid-dir", default="/data/eval/act_valid")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    L = a.layer
    out_dir = Path(a.out or f"/data/eval/recon_L{L}_{a.init}{'_frozen' if a.freeze_codes else ''}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = {"args": vars(a)}

    # ---- data (BF16 residual stream from llama.cpp dumps)
    tdirs = [Path(d) for d in a.train_dir.split(",")]
    xtr = torch.cat([load_acts(d, f"l_out-{L-1}") for d in tdirs])
    ytr = torch.cat([load_acts(d, f"l_out-{L}") for d in tdirs])
    xva, yva = load_acts(Path(a.valid_dir), f"l_out-{L-1}"), load_acts(Path(a.valid_dir), f"l_out-{L}")
    print(f"train {tuple(xtr.shape)}  valid {tuple(xva.shape)}")

    # ---- teacher, and a sanity check that our PyTorch layer reproduces llama.cpp's BF16 layer
    w_base = load_layer_weights(L)
    teacher, cfg = build_layer(L, w_base)
    tm = all_metrics(teacher, xva, yva, cfg)
    print(f"teacher(pytorch) vs llama.cpp BF16 l_out-{L}: full cos {tm['full']['cos_mean']:.6f} "
          f"relMSE {tm['full']['rel_mse']:.2e} | delta cos {tm['delta']['cos_mean']:.5f} relMSE {tm['delta']['rel_mse']:.2e}")
    log["teacher_vs_llamacpp"] = tm
    # use the PyTorch teacher's outputs as targets so implementation noise cancels
    with torch.no_grad():
        ytr_t = torch.cat([layer_forward(teacher, xtr[i:i + 4].cuda(), cfg).float().cpu() for i in range(0, xtr.shape[0], 4)])
        yva_t = torch.cat([layer_forward(teacher, xva[i:i + 4].cuda(), cfg).float().cpu() for i in range(0, xva.shape[0], 4)])
    del teacher; torch.cuda.empty_cache()

    # ---- shipped Bonsai 2 layer as the bar
    w_ship = shipped_layer_weights(L)
    ship_codes = {n: torch.sign(w_ship[n + ".weight"]).to(torch.int8) for n in TERNARY_LINEARS if n + ".weight" in w_ship}
    shipped, _ = build_layer(L, {**w_base, **{k: v for k, v in w_ship.items() if not any(k.startswith(n) for n in TERNARY_LINEARS)}})
    signs = shipped_signs()
    for n in TERNARY_LINEARS:
        if n + ".weight" in w_ship:
            parent, _, child = n.rpartition(".")
            wf = w_ship[n + ".weight"].cuda()
            setattr(shipped.get_submodule(parent), child, TernaryLinear.from_folded(wf, signs[wf.shape[1]].cuda()).cuda())
    sm = all_metrics(shipped, xva, yva_t, cfg)
    sc = code_agreement(shipped, ship_codes)
    print(f"SHIPPED layer: full cos {sm['full']['cos_mean']:.5f} relMSE {sm['full']['rel_mse']:.4f} | "
          f"delta cos {sm['delta']['cos_mean']:.4f} relMSE {sm['delta']['rel_mse']:.4f}   (self code agree {min(v['agree'] for v in sc.values()):.4f})")
    log["shipped"] = {"metrics": sm}
    if a.init != "shipped":
        del shipped; torch.cuda.empty_cache()

    # ---- student
    if a.init == "shipped":
        student = shipped
    else:
        student, _ = build_layer(L, w_base)
        student = ternarize_layer(student, init=a.init)
    init_codes = student_codes(student)
    m0 = all_metrics(student, xva, yva_t, cfg)
    c0 = code_agreement(student, ship_codes)
    print(f"INIT({a.init}): full cos {m0['full']['cos_mean']:.5f} relMSE {m0['full']['rel_mse']:.4f} | "
          f"delta cos {m0['delta']['cos_mean']:.4f} relMSE {m0['delta']['rel_mse']:.4f}")
    print("  code agreement with shipped:", {GGUF_NAME[k]: round(v["agree"], 4) for k, v in c0.items()})
    log["init"] = {"metrics": m0, "code_vs_shipped": c0}

    if a.freeze_codes:
        for n in TERNARY_LINEARS:
            if n in init_codes:
                student.get_submodule(n).frozen_codes = init_codes[n].cuda()

    # ---- optimiser: latent weights / scales / norms
    p_w, p_s, p_n = [], [], []
    for name, p in student.named_parameters():
        if isinstance(student.get_submodule(name.rpartition(".")[0]), TernaryLinear):
            (p_s if name.endswith("scale") else p_w).append(p)
        elif "norm" in name:
            p_n.append(p)
        else:
            p.requires_grad_(False)
    if a.freeze_codes:
        for p in p_w: p.requires_grad_(False)
        p_w = []
    print(f"trainable: latent {sum(p.numel() for p in p_w)/1e6:.1f}M, scales {sum(p.numel() for p in p_s)/1e6:.2f}M, norms {sum(p.numel() for p in p_n)}")
    groups = [g for g in ({"params": p_w, "lr": a.lr_w}, {"params": p_s, "lr": a.lr_s}, {"params": p_n, "lr": a.lr_norm}) if g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(t, a.steps) / a.steps)))

    # ---- train
    n_tr = xtr.shape[0]
    g = torch.Generator().manual_seed(0)
    hist = []
    t0 = time.time()
    student.train()
    for step in range(1, a.steps + 1):
        idx = torch.randperm(n_tr, generator=g)[: a.bs]
        x = xtr[idx].cuda(); y_ref = ytr_t[idx].cuda()
        y = layer_forward(student, x, cfg).float()
        d, d_ref = y - x, y_ref - x
        l_mse = ((d - d_ref) ** 2).sum() / (d_ref ** 2).sum()
        l_cos = 1 - F.cosine_similarity(y.reshape(-1, y.shape[-1]), y_ref.reshape(-1, y.shape[-1]), dim=1).mean()
        loss = l_mse + a.cos_weight * l_cos
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(p_w + p_s + p_n, 1.0)
        opt.step(); sched.step()
        if step % 20 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  delta_relmse {l_mse.item():.4f}  1-cos {l_cos.item():.5f}  ({time.time()-t0:.0f}s)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            m = all_metrics(student, xva, yva_t, cfg); student.train()
            c = code_agreement(student, ship_codes); ci = code_agreement(student, init_codes)
            hist.append({"step": step, "metrics": m, "code_vs_shipped": c, "code_vs_init": ci})
            print(f"  [eval {step}] full cos {m['full']['cos_mean']:.5f} relMSE {m['full']['rel_mse']:.4f} | delta cos {m['delta']['cos_mean']:.4f} "
                  f"relMSE {m['delta']['rel_mse']:.4f} | code agree shipped {sum(v['agree'] for v in c.values())/len(c):.4f} "
                  f"init {sum(v['agree'] for v in ci.values())/len(ci):.4f}", flush=True)
    log["history"] = hist
    log["final"] = hist[-1] if hist else None
    torch.save({n: student.get_submodule(n).codes().cpu() for n in init_codes}, out_dir / "codes.pt")
    torch.save({k: v.detach().cpu() for k, v in student.state_dict().items() if not k.endswith((".H", ".signs"))},
               out_dir / "student_state.pt")
    torch.save({n: student.get_submodule(n).scale.detach().cpu() for n in init_codes}, out_dir / "scales.pt")
    (out_dir / "log.json").write_text(json.dumps(log, indent=1))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
