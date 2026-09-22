"""Joint logit distillation of the last student blocks + final norm + lm_head.

    python train_joint.py --first 60 --steps 600 --head lora

student : ternary layers first..63 (restored from the progressive run; latent re-initialised
          at the ternary points, so codes can keep moving) + final RMSNorm + lm_head (+LoRA)
input   : student stream S_first (regenerated), teacher logits = BF16 head on canonical C_64
loss    : KL(teacher || student) * T^2 + lam * CE(ground truth) + anchor * relMSE(stream_64, C_64)
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
import torch.utils.checkpoint as ck

sys.path.insert(0, str(Path(__file__).resolve().parent))
from head_calib import D, Head, bf16_head, eval_combo  # noqa: E402
from progressive import ACT_DIRS, VALID_DIR, Rotary, fwd, load_f16, tokens_of  # noqa: E402
from student_utils import load_student_layer  # noqa: E402
from ternary_layer import TERNARY_LINEARS, TernaryLinear, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=60)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr-w", type=float, default=1e-4)
    ap.add_argument("--lr-s", type=float, default=5e-5)
    ap.add_argument("--lr-norm", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--head", default="lora", choices=["none", "norm", "diag", "lora"])
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--anchor", type=float, default=0.1)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--freeze-codes", action="store_true")
    ap.add_argument("--no-bias", action="store_true", help="no vocab bias (GGUF cannot store one)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--dir", default="/data/eval/prog_dual")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True; torch.manual_seed(0)
    d = Path(a.dir); cfg = load_config(); rotary = Rotary(cfg)
    out = Path(a.out or f"/data/eval/joint_L{a.first}_{a.head}{'_frozen' if a.freeze_codes else ''}"); out.mkdir(parents=True, exist_ok=True)

    tok_tr = torch.cat([tokens_of(x) for x in ACT_DIRS]); tok_va = tokens_of(VALID_DIR)
    S_tr, S_va = load_f16(d / f"S_{a.first}_train.f16"), load_f16(d / f"S_{a.first}_valid.f16")
    C_tr, C_va = load_f16(d / "C_64_train.f16"), load_f16(d / "C_64_valid.f16")
    ref = bf16_head()
    head = bf16_head()
    if a.head == "lora": head.add_lora(a.rank)
    layers = [load_student_layer(k, d, trainable_latent=not a.freeze_codes)[0] for k in range(a.first, 64)]

    def block(x):
        h = x
        for m in layers:
            h = ck.checkpoint(lambda inp, m=m: fwd(m, inp, rotary), h, use_reentrant=False) if torch.is_grad_enabled() else fwd(m, h, rotary)
        return h

    @torch.no_grad()
    def evaluate():
        for m in layers: m.eval()
        hs = torch.cat([block(S_va[i:i + 4].cuda().float()).to(torch.float16).cpu() for i in range(0, S_va.shape[0], 4)])
        m = eval_combo(hs, head, ref, C_va, tok_va)
        for mm in layers: mm.train()
        return m

    m0 = evaluate(); print(f"before: valid KL {m0['kl']:.4f} top-1 {m0['top1']:.4f} PPL {m0['ppl']:.3f}", flush=True)

    p_w, p_s, p_n, p_h = [], [], [], []
    for m in layers:
        for name, p in m.named_parameters():
            if isinstance(m.get_submodule(name.rpartition(".")[0]), TernaryLinear):
                (p_s if name.endswith("scale") else p_w).append(p)
            elif "norm" in name:
                p_n.append(p)
            else:
                p.requires_grad_(False)
    if a.freeze_codes:
        for p in p_w: p.requires_grad_(False)
        p_w = []
    for p in head.parameters(): p.requires_grad_(False)
    if a.head != "none":
        p_n.append(head.norm); head.norm.requires_grad_(True)
    if a.head in ("diag", "lora"):
        p_h += [head.in_scale] + ([] if a.no_bias else [head.bias])
    if a.head == "lora":
        p_h += [head.lora_a, head.lora_b]
    for p in p_h: p.requires_grad_(True)
    groups = [g for g in ({"params": p_w, "lr": a.lr_w}, {"params": p_s, "lr": a.lr_s}, {"params": p_n, "lr": a.lr_norm}, {"params": p_h, "lr": a.lr_head}) if g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(t, a.steps) / a.steps)))
    print(f"trainable: latent {sum(p.numel() for p in p_w)/1e6:.0f}M scales {sum(p.numel() for p in p_s)/1e6:.1f}M norms {sum(p.numel() for p in p_n)} head {sum(p.numel() for p in p_h)/1e6:.2f}M", flush=True)

    n = S_tr.shape[0]; g = torch.Generator().manual_seed(0); T = a.temp; hist = []; t0 = time.time()
    for m in layers: m.train()
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            idx = torch.randperm(n, generator=g)[: a.bs]
            x = S_tr[idx].cuda().float(); c = C_tr[idx].cuda().float(); tok = tok_tr[idx].cuda()
            with torch.no_grad():
                lt = F.log_softmax(ref(c.reshape(-1, D)) / T, -1)
            h = block(x)
            s = head(h.reshape(-1, D))
            ls = F.log_softmax(s / T, -1)
            kl = (lt.exp() * (lt - ls)).sum(-1).mean() * T * T
            L = x.shape[1]
            ce = F.cross_entropy(s.reshape(-1, L, s.shape[-1])[:, :-1].reshape(-1, s.shape[-1]), tok[:, 1:].reshape(-1))
            anc = ((h - c) ** 2).sum() / ((c - x) ** 2).sum().clamp(min=1e-6) if a.anchor > 0 else torch.zeros((), device=x.device)
            loss = (kl + a.lam * ce + a.anchor * anc) / a.accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(p_w + p_s + p_n + p_h, 1.0)
        opt.step(); sched.step()
        if step % 25 == 0:
            print(f"  step {step:5d} kl {kl.item():.4f} ce {ce.item():.4f} anc {anc.item():.4f} ({time.time()-t0:.0f}s) mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            m = evaluate(); hist.append({"step": step, **m})
            print(f"  [eval {step}] valid KL {m['kl']:.4f} top-1 {m['top1']:.4f} PPL {m['ppl']:.3f}", flush=True)
    for k, m in zip(range(a.first, 64), layers):
        sd = {}
        for nme in TERNARY_LINEARS:
            try: tl = m.get_submodule(nme)
            except AttributeError: continue
            if isinstance(tl, TernaryLinear):
                sd[nme + ".codes"] = tl.codes().cpu(); sd[nme + ".scale"] = tl.scale.detach().cpu()
        for kk, v in m.named_parameters():
            if "norm" in kk: sd[kk] = v.detach().cpu()
        torch.save(sd, out / f"student_compact_L{k}.pt")
    torch.save({k: v.detach().cpu() for k, v in head.named_parameters()}, out / "head_params.pt")
    json.dump({"args": vars(a), "before": m0, "history": hist}, open(out / "log.json", "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
