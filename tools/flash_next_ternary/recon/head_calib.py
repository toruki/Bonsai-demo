"""Head cross-evaluation and calibration for the progressive ternary 27B.

Streams (final residual, layer-63 exit) x heads (final RMSNorm + lm_head):
    streams: bf16 (canonical), prog (progressive student), ship (shipped Bonsai 2)
    heads  : bf16 (base norm + BF16 lm_head), ship (shipped norm + rotated ternary lm_head)
Metrics vs the BF16-stream/BF16-head logits: KL(teacher||student), top-1 agreement,
and next-token CE (perplexity) against the ground-truth tokens.

Calibration (`--train MODE`): student stream fixed (prog S_64), teacher = BF16 logits,
    L = KL(softmax(t/T) || softmax(s/T)) * T^2 + lam * CE(ground truth)
    MODE = norm      : final RMSNorm weight only
           norm+diag : + per-input-channel scale on lm_head input and per-vocab bias
           lora      : + LoRA (rank r) on lm_head
           full      : + full lm_head
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bonsai_format import PTQ1_0_BLOCK_BYTES, hadamard_matrix, ptq1_0_dequantize  # noqa: E402
from progressive import load_f16, tokens_of  # noqa: E402
from ternary_layer import BASE, BLOCK, SHIPPED_PTQ, load_config, shipped_signs  # noqa: E402

D = 5120


# ------------------------------------------------------------------ heads

class Head(nn.Module):
    """final RMSNorm (multiplier form) -> optional rotation -> lm_head; with calibration knobs."""

    def __init__(self, norm_mult: torch.Tensor, W: torch.Tensor, rotate: bool, signs=None, eps=1e-6):
        super().__init__()
        self.norm = nn.Parameter(norm_mult.float().clone())          # multiplier (1+w) form
        self.register_buffer("W", W)                                  # [V, D] (bf16 or f16)
        self.rotate = rotate
        if rotate:
            self.register_buffer("signs", signs.float())
            self.register_buffer("H", torch.from_numpy(hadamard_matrix(BLOCK)))
        self.eps = eps
        # calibration knobs (identity by default)
        self.in_scale = nn.Parameter(torch.ones(D))
        self.bias = nn.Parameter(torch.zeros(W.shape[0]))
        self.lora_a = None; self.lora_b = None
        self.W_delta = None

    def add_lora(self, r: int):
        V = self.W.shape[0]
        self.lora_a = nn.Parameter(torch.randn(r, D, device=self.W.device) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(V, r, device=self.W.device))

    def add_full(self):
        self.W_delta = nn.Parameter(torch.zeros_like(self.W, dtype=torch.float32))

    def forward(self, h: torch.Tensor) -> torch.Tensor:              # h [N, D] f32 -> logits [N, V] f32
        x = h.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.norm
        if self.rotate:
            x = ((x * self.signs).reshape(-1, BLOCK) @ self.H).reshape(-1, D)
        x = x * self.in_scale
        W = self.W.float() if self.W_delta is None else self.W.float() + self.W_delta
        y = x @ W.t()
        if self.lora_a is not None:
            y = y + (x @ self.lora_a.t()) @ self.lora_b.t()
        return y + self.bias


def bf16_head(device="cuda") -> Head:
    from safetensors import safe_open
    index = json.load(open(BASE / "model.safetensors.index.json"))["weight_map"]
    with safe_open(str(BASE / index["lm_head.weight"]), "pt") as f:
        W = f.get_tensor("lm_head.weight")
    with safe_open(str(BASE / index["model.language_model.norm.weight"]), "pt") as f:
        nw = f.get_tensor("model.language_model.norm.weight").float() + 1.0      # HF stores w-1
    return Head(nw, W.to(device), rotate=False).to(device)


def shipped_head(device="cuda") -> Head:
    from gguf import GGUFReader
    r = GGUFReader(str(SHIPPED_PTQ)); by = {t.name: t for t in r.tensors}
    t = by["output.weight"]; ne0 = int(t.shape[0])
    raw = t.data.view(np.uint8).reshape(-1, PTQ1_0_BLOCK_BYTES)
    Wf = torch.from_numpy(ptq1_0_dequantize(raw).reshape(-1, ne0).copy()).to(torch.float16)
    nw = torch.from_numpy(np.array(by["output_norm.weight"].data, dtype=np.float32).reshape(-1))
    return Head(nw, Wf.to(device), rotate=True, signs=shipped_signs()[D]).to(device)


# ---------------------------------------------------------------- streams

def load_llamacpp_stream(d: str, name="l_out-63") -> torch.Tensor:
    m = json.load(open(Path(d) / "meta.json")); ne0, nt = m["tensors"][name]
    x = np.fromfile(Path(d) / f"{name}.f32", dtype=np.float32).reshape(nt, ne0)
    return torch.from_numpy(x).reshape(m["n_sequences"], m["seq_len"], ne0).to(torch.float16)


# ------------------------------------------------------------------ eval

@torch.no_grad()
def eval_combo(stream: torch.Tensor, head: Head, ref_head: Head, ref_stream: torch.Tensor,
               tokens: torch.Tensor, bs=2) -> dict:
    """KL(ref || this), top-1 agreement, and next-token CE, over all sequences."""
    head.eval(); ref_head.eval()
    kl_sum = 0.0; agree = 0; ce_sum = 0.0; n = 0
    for i in range(0, stream.shape[0], bs):
        h = stream[i:i + bs].cuda().float().reshape(-1, D); hr = ref_stream[i:i + bs].cuda().float().reshape(-1, D)
        tok = tokens[i:i + bs].cuda().reshape(-1)
        L = stream.shape[1]
        s = head(h); t = ref_head(hr)
        ls, lt = F.log_softmax(s, -1), F.log_softmax(t, -1)
        kl = (lt.exp() * (lt - ls)).sum(-1)
        # next-token CE: position j predicts tok[j+1] within each sequence
        ls_ = ls.reshape(-1, L, ls.shape[-1])[:, :-1].reshape(-1, ls.shape[-1])
        nxt = tokens[i:i + bs].cuda()[:, 1:].reshape(-1)
        ce = F.nll_loss(ls_, nxt, reduction="sum")
        kl_sum += kl.sum().item(); agree += (s.argmax(-1) == t.argmax(-1)).sum().item(); n += kl.numel()
        ce_sum += ce.item()
        del s, t, ls, lt
    n_ce = stream.shape[0] * (stream.shape[1] - 1)
    return {"kl": kl_sum / n, "top1": agree / n, "ppl": math.exp(ce_sum / n_ce)}


# ------------------------------------------------------------------ train

def train_head(head: Head, ref_head: Head, S_tr, C_tr, tok_tr, S_va, C_va, tok_va, a) -> list:
    T = a.temp
    params = [head.norm]
    if a.train in ("norm+diag", "lora", "full"):
        params += [head.in_scale, head.bias]
    if a.train == "lora":
        head.add_lora(a.rank); params += [head.lora_a, head.lora_b]
    if a.train == "full":
        head.add_full(); params += [head.W_delta]
    for p in head.parameters(): p.requires_grad_(False)
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW([{"params": params, "lr": a.lr}], betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(t, a.steps) / a.steps)))
    print(f"trainable: {sum(p.numel() for p in params)/1e6:.3f}M")
    n = S_tr.shape[0]; g = torch.Generator().manual_seed(0); hist = []; t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = torch.randperm(n, generator=g)[: a.bs]
        h = S_tr[idx].cuda().float().reshape(-1, D); hr = C_tr[idx].cuda().float().reshape(-1, D)
        with torch.no_grad():
            lt = F.log_softmax(ref_head(hr) / T, -1)
        s = head(h)
        ls = F.log_softmax(s / T, -1)
        kl = (lt.exp() * (lt - ls)).sum(-1).mean() * T * T
        L = S_tr.shape[1]
        ce = F.cross_entropy(s.reshape(-1, L, s.shape[-1])[:, :-1].reshape(-1, s.shape[-1]), tok_tr[idx].cuda()[:, 1:].reshape(-1))
        loss = kl + a.lam * ce
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 50 == 0:
            print(f"  step {step:5d} kl {kl.item():.4f} ce {ce.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            m = eval_combo(S_va, head, ref_head, C_va, tok_va); hist.append({"step": step, **m})
            print(f"  [eval {step}] valid KL {m['kl']:.4f} top-1 {m['top1']:.4f} PPL {m['ppl']:.3f}", flush=True)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross-eval", action="store_true")
    ap.add_argument("--train", default=None, choices=["norm", "norm+diag", "lora", "full"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--prog-dir", default="/data/eval/prog_dual")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)

    tok_va = tokens_of("/data/eval/act_valid")
    C_va = load_f16(Path(a.prog_dir) / "C_64_valid.f16"); S_va = load_f16(Path(a.prog_dir) / "S_64_valid.f16")
    Hb = bf16_head()

    if a.cross_eval:
        Hs = shipped_head()
        ship_va = load_llamacpp_stream("/data/eval/ship_final_valid")
        # sanity: canonical stream vs llama.cpp BF16 final stream (if available)
        res = {}
        combos = [("bf16 stream -> bf16 head", C_va, Hb), ("prog stream -> bf16 head", S_va, Hb),
                  ("ship stream -> ship head", ship_va, Hs), ("ship stream -> bf16 head", ship_va, Hb),
                  ("prog stream -> ship head", S_va, Hs), ("bf16 stream -> ship head", C_va, Hs)]
        print(f"{'combo':30s} {'KL vs BF16':>11s} {'top-1':>7s} {'PPL(valid)':>11s}")
        for name, st, hd in combos:
            m = eval_combo(st, hd, Hb, C_va, tok_va); res[name] = m
            print(f"{name:30s} {m['kl']:11.4f} {m['top1']:7.4f} {m['ppl']:11.3f}")
        json.dump(res, open("/data/eval/head_cross_eval.json", "w"), indent=1)

    if a.train:
        tok_tr = torch.cat([tokens_of(d) for d in ("/data/eval/act_train", "/data/eval/act_train2")])
        C_tr = load_f16(Path(a.prog_dir) / "C_64_train.f16"); S_tr = load_f16(Path(a.prog_dir) / "S_64_train.f16")
        head = bf16_head()
        m0 = eval_combo(S_va, head, Hb, C_va, tok_va)
        print(f"before: valid KL {m0['kl']:.4f} top-1 {m0['top1']:.4f} PPL {m0['ppl']:.3f}")
        hist = train_head(head, Hb, S_tr, C_tr, tok_tr, S_va, C_va, tok_va, a)
        out = Path(a.out or f"/data/eval/head_{a.train}"); out.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in head.named_parameters()}, out / "head_params.pt")
        json.dump({"args": vars(a), "before": m0, "history": hist}, open(out / "log.json", "w"), indent=1)
        print("wrote", out)


if __name__ == "__main__":
    main()
