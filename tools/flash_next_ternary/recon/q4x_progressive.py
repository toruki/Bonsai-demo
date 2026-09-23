"""Progressive ternary-aware reconstruction of Qwen3.8-Flash-Next MoE experts.

Per layer L (block = 1 layer, because one layer's experts are 2.5 B params):
    S_L  student hyper stream entering L   (S_first = canonical, earlier layers untouched)
    C_L  canonical teacher stream entering L
    teacher = the IQ3_XXS GGUF layer, dequantized (checked against llama.cpp to ~1e-6)
    A    = teacher(S_L)          same-input target
    C    = teacher(C_L)          canonical target (anchor)
    student = teacher layer with routed experts replaced by Hadamard-folded ternary
              experts (H128, group-128 scales, codes free to move via STE); routing is
              computed from the student's own stream every step (router frozen)
    loss = relMSE_delta(y, A) + anchor * relMSE_delta(y, C) + (1 - cos(y, A))
Trainable: expert latents (Adafactor, factored, no momentum -- AdamW state would not fit),
group scales and the two hyper-connection norms (AdamW).
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from bonsai_format import hadamard_matrix  # noqa: E402
from gguf_inject_ternary import make_signs  # noqa: E402
from q4x_layer import FLASH_GGUF, GGUFTensors, LayerRunner, build_layer, layer_weights, text_config  # noqa: E402

BLOCK = 128
GROUP = 128


# ------------------------------------------------------------ rotation helpers

class Rot:
    """x -> H_blk (s ⊙ x) along the last axis, and its transpose (= inverse)."""

    def __init__(self, width: int, device):
        self.s = torch.from_numpy(make_signs(width, 0).astype(np.float32)).to(device)
        self.H = torch.from_numpy(hadamard_matrix(BLOCK)).to(device)
        self.w = width

    def fwd(self, x):
        return ((x * self.s).reshape(-1, BLOCK) @ self.H).reshape(x.shape)

    def bwd(self, g):                       # transpose: s ⊙ (H^T g), H symmetric
        return (g.reshape(-1, BLOCK) @ self.H).reshape(g.shape) * self.s

    def fold(self, w):                       # W -> (W * s) @ H along the last axis
        return ((w * self.s).reshape(-1, BLOCK) @ self.H).reshape(w.shape)


def mseopt_scale(wf: torch.Tensor) -> torch.Tensor:
    """exact per-group weight-MSE-optimal ternary scale; wf [..., G*k] -> [..., k]."""
    g = wf.reshape(-1, GROUP).abs()
    srt, _ = torch.sort(g, dim=1, descending=True)
    cs = torch.cumsum(srt, dim=1)
    k = torch.arange(1, GROUP + 1, device=g.device, dtype=g.dtype)
    kb = (cs * cs / k).argmax(dim=1)
    s = cs.gather(1, kb[:, None])[:, 0] / (kb + 1).to(g.dtype)
    return s.clamp(min=1e-8).reshape(*wf.shape[:-1], wf.shape[-1] // GROUP)


def quant(lat, s):
    """value of the ternary weight (no autograd); lat [..., n], s [..., n/GROUP]."""
    u = lat.reshape(*lat.shape[:-1], -1, GROUP) / s[..., None]
    t = torch.clamp(torch.round(u), -1, 1)
    return (t * s[..., None]).reshape(lat.shape), t, u


def ste_grads(gW, lat, s):
    """grads of the STE quantiser: d/dlat = inside; d/ds = sum_group gW * (t - inside*u)."""
    _, t, u = quant(lat, s)
    inside = (u.abs() <= 1.5).to(gW.dtype)
    g = gW.reshape(*gW.shape[:-1], -1, GROUP)
    return (g * inside).reshape(gW.shape), (g * (t - inside * u)).sum(-1)


# ------------------------------------------------------- fused ternary MoE op

class FusedAdam:
    """Adam applied per expert inside TernaryMoE.backward: the full-size latent gradient
    (10 GB) is never materialised, and m/v are kept in bf16 (10 GB instead of 20)."""

    def __init__(self, params: dict, lr=2e-4, betas=(0.9, 0.99), eps=1e-8):
        self.m = {k: torch.zeros_like(p, dtype=torch.bfloat16) for k, p in params.items()}
        self.v = {k: torch.zeros_like(p, dtype=torch.bfloat16) for k, p in params.items()}
        self.lr, self.b1, self.b2, self.eps, self.t = lr, betas[0], betas[1], eps, 0

    @torch.no_grad()
    def update(self, key, lat_e, e, grad):
        m, v = self.m[key][e], self.v[key][e]
        mf = m.float().mul_(self.b1).add_(grad, alpha=1 - self.b1)
        vf = v.float().mul_(self.b2).addcmul_(grad, grad, value=1 - self.b2)
        m.copy_(mf); v.copy_(vf)
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        lat_e.addcdiv_(mf, vf.div_(bc2).sqrt_().add_(self.eps), value=-self.lr / bc1)


class TernaryMoE(torch.autograd.Function):
    """All routed experts in one op, so each expert's quantised weight is recomputed
    instead of saved, and parameter grads are materialised once (not per expert)."""

    @staticmethod
    def forward(ctx, x_rot, idx, wts, gu_lat, gu_s, dn_lat, dn_s, rot_i, I, opt):
        T = x_rot.shape[0]
        out = torch.zeros(T, dn_lat.shape[1], device=x_rot.device, dtype=x_rot.dtype)
        E = gu_lat.shape[0]
        mask = F.one_hot(idx, num_classes=E).permute(2, 1, 0)          # [E, k, T]
        hit = torch.nonzero(mask.sum(dim=(1, 2)) > 0)[:, 0].tolist()
        for e in hit:
            pos, ti = torch.where(mask[e])
            xe = x_rot[ti]
            gu = xe @ quant(gu_lat[e], gu_s[e])[0].t()
            g, u = gu[:, :I], gu[:, I:]
            h = F.silu(g) * u
            ye = rot_i.fwd(h) @ quant(dn_lat[e], dn_s[e])[0].t()
            out.index_add_(0, ti, ye * wts[ti, pos, None])
        ctx.save_for_backward(x_rot, idx, wts, gu_lat, gu_s, dn_lat, dn_s)
        ctx.rot_i, ctx.I, ctx.hit, ctx.mask, ctx.opt = rot_i, I, hit, mask, opt
        return out

    @staticmethod
    def backward(ctx, gout):
        x_rot, idx, wts, gu_lat, gu_s, dn_lat, dn_s = ctx.saved_tensors
        rot_i, I = ctx.rot_i, ctx.I
        gx = torch.zeros_like(x_rot)
        gw = torch.zeros_like(wts)
        opt = ctx.opt
        g_gul = None if opt is not None else torch.zeros_like(gu_lat)
        g_dnl = None if opt is not None else torch.zeros_like(dn_lat)
        g_gus, g_dns = torch.zeros_like(gu_s), torch.zeros_like(dn_s)
        for e in ctx.hit:
            pos, ti = torch.where(ctx.mask[e])
            xe = x_rot[ti]
            Wgu = quant(gu_lat[e], gu_s[e])[0]
            Wdn = quant(dn_lat[e], dn_s[e])[0]
            gu = xe @ Wgu.t()
            g, u = gu[:, :I], gu[:, I:]
            sg = F.silu(g)
            h = sg * u
            hr = rot_i.fwd(h)
            ye = hr @ Wdn.t()
            go = gout[ti]
            w = wts[ti, pos, None]
            gw[ti, pos] = (go * ye).sum(-1)
            gye = go * w
            gWdn = gye.t() @ hr
            a, b = ste_grads(gWdn, dn_lat[e], dn_s[e]); g_dns[e] += b
            gh = rot_i.bwd(gye @ Wdn)                     # uses Wdn before the latent moves
            if opt is None: g_dnl[e] += a
            else: opt.update("dn", dn_lat[e], e, a)
            sig = torch.sigmoid(g)
            gg = gh * u * (sig * (1 + g * (1 - sig)))
            gu_ = gh * sg
            ggu = torch.cat([gg, gu_], dim=1)
            gWgu = ggu.t() @ xe
            a, b = ste_grads(gWgu, gu_lat[e], gu_s[e]); g_gus[e] += b
            gx.index_add_(0, ti, ggu @ Wgu)               # uses Wgu before the latent moves
            if opt is None: g_gul[e] += a
            else: opt.update("gu", gu_lat[e], e, a)
        return gx, None, gw, g_gul, g_gus, g_dnl, g_dns, None, None, None


class TernaryExperts(nn.Module):
    """Drop-in for Qwen4ExpTextExperts: folded ternary experts with learnable scales."""

    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor, device):
        super().__init__()
        E, twoI, H = gate_up.shape
        self.I = twoI // 2
        self.rot_h, self.rot_i = Rot(H, device), Rot(self.I, device)
        gl, gs, dl, ds = [], [], [], []
        with torch.no_grad():
            for e0 in range(0, E, 32):                       # fold + scale init in chunks
                gf = self.rot_h.fold(gate_up[e0:e0 + 32].to(device))
                df = self.rot_i.fold(down[e0:e0 + 32].to(device))
                gl.append(gf); gs.append(mseopt_scale(gf)); dl.append(df); ds.append(mseopt_scale(df))
        self.gu_lat = nn.Parameter(torch.cat(gl)); self.gu_s = nn.Parameter(torch.cat(gs))
        self.dn_lat = nn.Parameter(torch.cat(dl)); self.dn_s = nn.Parameter(torch.cat(ds))
        self.opt = None                                          # set to FusedAdam to train

    def forward(self, hidden_states, top_k_index, top_k_weights):
        x_rot = self.rot_h.fwd(hidden_states)
        return TernaryMoE.apply(x_rot, top_k_index, top_k_weights, self.gu_lat, self.gu_s,
                                self.dn_lat, self.dn_s, self.rot_i, self.I, self.opt)

    @torch.no_grad()
    def export(self):
        """folded ternary values per GGUF tensor ([E, rows, in] numpy float32) + codes."""
        gu, gt, _ = quant(self.gu_lat, self.gu_s.half().float())
        dn, dt, _ = quant(self.dn_lat, self.dn_s.half().float())
        I = self.I
        return {"gate": gu[:, :I].cpu().numpy(), "up": gu[:, I:].cpu().numpy(), "down": dn.cpu().numpy(),
                "codes": {"gate": gt[:, :I].to(torch.int8).cpu(), "up": gt[:, I:].to(torch.int8).cpu(),
                          "down": dt.to(torch.int8).cpu()}}


# --------------------------------------------------------------------- data

def load_stream(d: Path, width: int) -> torch.Tensor:
    m = json.load(open(d / "meta.json"))
    x = np.fromfile(d / "l_last-3.f32", dtype=np.float32).reshape(m["n_sequences"], m["seq_len"], width)
    return torch.from_numpy(x).to(torch.float16)


@torch.no_grad()
def run(mod, runner, X, bs=2, dev="cuda"):
    return torch.cat([runner(mod, X[i:i + bs].to(dev).float()).to(torch.float16).cpu() for i in range(0, X.shape[0], bs)])


def rel(a, b):
    return float(((a.float() - b.float()) ** 2).sum() / (b.float() ** 2).sum().clamp(min=1e-12))


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=4)
    ap.add_argument("--last", type=int, default=11)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr-w", type=float, default=2e-4)
    ap.add_argument("--lr-s", type=float, default=1e-4)
    ap.add_argument("--lr-norm", type=float, default=1e-4)
    ap.add_argument("--anchor", type=float, default=0.2)
    ap.add_argument("--cos-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--train-dir", default="/data/eval/q4x_act_train")
    ap.add_argument("--valid-dir", default="/data/eval/q4x_act_valid")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    dev = "cuda"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = text_config(); width = cfg.hidden_size * cfg.hc_count
    runner = LayerRunner(cfg, dev)
    g = GGUFTensors(FLASH_GGUF)
    S_tr = load_stream(Path(a.train_dir), width); S_va = load_stream(Path(a.valid_dir), width)
    C_tr, C_va = S_tr.clone(), S_va.clone()
    print(f"streams: train {tuple(S_tr.shape)} valid {tuple(S_va.shape)}", flush=True)
    log = {"args": vars(a), "layers": {}}

    for L in range(a.first, a.last + 1):
        t0 = time.time(); ll = {}
        W = layer_weights(g, L, cfg)
        teacher, _, miss, unexp = build_layer(L, W, device=dev)
        assert not miss and not unexp
        A_tr, A_va = run(teacher, runner, S_tr), run(teacher, runner, S_va)
        Cx_tr, Cx_va = run(teacher, runner, C_tr), run(teacher, runner, C_va)
        del teacher; torch.cuda.empty_cache()
        print(f"\n=== layer {L} ({cfg.layer_types[L]}) teachers {time.time()-t0:.0f}s", flush=True)

        student, _, _, _ = build_layer(L, {k: v for k, v in W.items() if not k.startswith("mlp.experts.")}, device=dev)
        student.mlp.experts = TernaryExperts(W["mlp.experts.gate_up_proj"], W["mlp.experts.down_proj"], dev)
        del W
        for p in student.parameters():
            p.requires_grad_(False)
        ex = student.mlp.experts
        scl = [ex.gu_s, ex.dn_s]
        nrm = [student.attn_hyper_connection.hc_norm.weight, student.mlp_hyper_connection.hc_norm.weight]
        for p in [ex.gu_lat, ex.dn_lat] + scl + nrm:
            p.requires_grad_(True)

        @torch.no_grad()
        def evaluate():
            Y = run(student, runner, S_va)
            dA = rel(Y.float() - S_va.float(), A_va.float() - S_va.float())
            return {"delta_vsA": dA, "stream_vsC": rel(Y, Cx_va), "Y": Y}
        e0 = evaluate(); ll["ptq"] = {k: v for k, v in e0.items() if k != "Y"}
        print(f"  PTQ init : delta vsA relMSE {e0['delta_vsA']:.4f} | stream vsC relMSE {e0['stream_vsC']:.4f}", flush=True)

        if a.steps > 0:
            ex.opt = FusedAdam({"gu": ex.gu_lat, "dn": ex.dn_lat}, lr=a.lr_w)
            opt_s = torch.optim.AdamW([{"params": scl, "lr": a.lr_s}, {"params": nrm, "lr": a.lr_norm}], betas=(0.9, 0.99), weight_decay=0.0)
            sched = lambda t: 0.5 * (1 + math.cos(math.pi * min(t, a.steps) / a.steps))  # noqa: E731
            gen = torch.Generator().manual_seed(L); tt = time.time()
            for step in range(1, a.steps + 1):
                f = sched(step)
                ex.opt.lr = a.lr_w * f; ex.opt.t = step
                for pg, base in zip(opt_s.param_groups, (a.lr_s, a.lr_norm)): pg["lr"] = base * f
                ix = torch.randperm(S_tr.shape[0], generator=gen)[: a.bs]
                x = S_tr[ix].to(dev).float(); yA = A_tr[ix].to(dev).float(); yC = Cx_tr[ix].to(dev).float()
                y = runner(student, x).float()
                d = y - x
                lA = ((d - (yA - x)) ** 2).sum() / ((yA - x) ** 2).sum()
                lC = ((d - (yC - x)) ** 2).sum() / ((yC - x) ** 2).sum()
                lcos = 1 - F.cosine_similarity(y.reshape(-1, width), yA.reshape(-1, width), dim=1).mean()
                loss = lA + a.anchor * lC + a.cos_weight * lcos
                opt_s.zero_grad(set_to_none=True)
                loss.backward()                           # latents are updated inside backward
                ex.gu_lat.grad = None; ex.dn_lat.grad = None
                torch.nn.utils.clip_grad_norm_(scl + nrm, 1.0)
                opt_s.step()
                with torch.no_grad():
                    for s in scl: s.clamp_(min=1e-6)
                if step % 50 == 0:
                    print(f"    step {step:4d} loss {loss.item():.4f} A {lA.item():.4f} C {lC.item():.4f} "
                          f"({time.time()-tt:.0f}s, peak {torch.cuda.max_memory_allocated()/2**30:.1f}G)", flush=True)
                if step % a.eval_every == 0 or step == a.steps:
                    ev = evaluate()
                    print(f"    [eval {step}] delta vsA relMSE {ev['delta_vsA']:.4f} | stream vsC relMSE {ev['stream_vsC']:.4f}", flush=True)
            ex.opt = None
            e1 = evaluate(); ll["trained"] = {k: v for k, v in e1.items() if k != "Y"}
        else:
            e1 = e0
        # advance streams
        S_tr = run(student, runner, S_tr); S_va = e1["Y"]
        C_tr, C_va = Cx_tr, Cx_va
        exp = ex.export()
        np.savez(out / f"experts_L{L}.npz", gate=exp["gate"].astype(np.float16), up=exp["up"].astype(np.float16),
                 down=exp["down"].astype(np.float16))
        ll["exit_stream_vsC"] = rel(S_va, C_va)
        ll["zero_frac"] = float(sum((c == 0).float().mean() for c in exp["codes"].values()) / 3)
        print(f"  EXIT layer {L}: stream vs canonical relMSE {ll['exit_stream_vsC']:.4f}  zero {ll['zero_frac']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        log["layers"][str(L)] = ll
        json.dump(log, open(out / "log.json", "w"), indent=1)
        del student, ex, scl, A_tr, A_va; torch.cuda.empty_cache()
    print("Q4X PROGRESSIVE DONE", flush=True)


if __name__ == "__main__":
    main()
