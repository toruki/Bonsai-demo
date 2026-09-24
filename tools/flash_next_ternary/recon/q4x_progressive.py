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
import gc
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
from ternary_store import load_experts, save_experts  # noqa: E402
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
    (10 GB) is never materialised, and m/v are kept in bf16 (10 GB instead of 20).

    state_bits=8 keeps them blockwise-quantised instead (5 GB): m as int8 and sqrt(v) as uint8,
    one fp16 absmax scale per GROUP elements. Every step still computes m, v and the update in
    fp32; only the state carried to the next step is rounded. A carried sqrt(v) that rounds to 0
    is read back as half a quantisation step, so a tiny v cannot turn into a huge update."""

    def __init__(self, params: dict, lr=2e-4, betas=(0.9, 0.99), eps=1e-8, state_bits=16):
        assert state_bits in (8, 16)
        self.bits = state_bits
        if state_bits == 16:
            self.m = {k: torch.zeros_like(p, dtype=torch.bfloat16) for k, p in params.items()}
            self.v = {k: torch.zeros_like(p, dtype=torch.bfloat16) for k, p in params.items()}
        else:
            sc = lambda p: torch.zeros(*p.shape[:-1], p.shape[-1] // GROUP, dtype=torch.float16, device=p.device)  # noqa: E731
            self.mq = {k: torch.zeros_like(p, dtype=torch.int8) for k, p in params.items()}
            self.vq = {k: torch.zeros_like(p, dtype=torch.uint8) for k, p in params.items()}
            self.ms = {k: sc(p) for k, p in params.items()}
            self.vs = {k: sc(p) for k, p in params.items()}
        self.lr, self.b1, self.b2, self.eps, self.t = lr, betas[0], betas[1], eps, 0

    @staticmethod
    def _blocks(x):
        return x.reshape(*x.shape[:-1], -1, GROUP)

    def _load(self, key, e):
        if self.bits == 16:
            return self.m[key][e].float(), self.v[key][e].float()
        shp = self.mq[key][e].shape
        m = (self._blocks(self.mq[key][e].float()) * self.ms[key][e].float()[..., None]).reshape(shp)
        q = self._blocks(self.vq[key][e].float())
        r = ((q + 0.5 * (q == 0)) * self.vs[key][e].float()[..., None]).reshape(shp)
        return m, r * r

    def _store(self, key, e, mf, vf):
        if self.bits == 16:
            self.m[key][e].copy_(mf); self.v[key][e].copy_(vf); return
        mb = self._blocks(mf)
        s = (mb.abs().amax(-1) / 127).to(torch.float16)
        sf = s.float()[..., None].clamp(min=1e-30)
        self.mq[key][e].copy_(torch.round(mb / sf).clamp_(-127, 127).reshape(mf.shape).to(torch.int8))
        self.ms[key][e].copy_(s)
        rb = self._blocks(vf.sqrt())
        s = (rb.amax(-1) / 255).to(torch.float16)
        sf = s.float()[..., None].clamp(min=1e-30)
        self.vq[key][e].copy_(torch.round(rb / sf).clamp_(0, 255).reshape(vf.shape).to(torch.uint8))
        self.vs[key][e].copy_(s)

    @torch.no_grad()
    def update(self, key, lat_e, e, grad):
        mf, vf = self._load(key, e)
        mf.mul_(self.b1).add_(grad, alpha=1 - self.b1)
        vf.mul_(self.b2).addcmul_(grad, grad, value=1 - self.b2)
        self._store(key, e, mf, vf)
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
    def load_values(self, src):
        """Set the latents to already-exported ternary values (experts_L{N}.npz, or a dict with the
        same gate/up/down arrays), so that quant(lat, s) reproduces them exactly:
        s = |value| of each group (1 for all-zero groups)."""
        z = load_experts(src) if isinstance(src, (str, Path)) else src
        for lat, sc, v in ((self.gu_lat, self.gu_s, np.concatenate([z["gate"], z["up"]], axis=1)),
                           (self.dn_lat, self.dn_s, z["down"])):
            for e0 in range(0, lat.shape[0], 32):
                w = torch.from_numpy(v[e0:e0 + 32]).to(lat.device).float()
                g = w.abs().reshape(*w.shape[:-1], -1, GROUP).amax(-1)
                lat[e0:e0 + 32] = w; sc[e0:e0 + 32] = torch.where(g > 0, g, torch.ones_like(g))
            del v

    @torch.no_grad()
    def export(self):
        """folded ternary values per GGUF tensor ([E, rows, in] numpy float32) + codes."""
        gu, gt, _ = quant(self.gu_lat, self.gu_s.half().float())
        dn, dt, _ = quant(self.dn_lat, self.dn_s.half().float())
        I = self.I
        return {"gate": gu[:, :I].cpu().numpy(), "up": gu[:, I:].cpu().numpy(), "down": dn.cpu().numpy(),
                "codes": {"gate": gt[:, :I].to(torch.int8).cpu(), "up": gt[:, I:].to(torch.int8).cpu(),
                          "down": dt.to(torch.int8).cpu()}}


class NextRouter:
    """Teacher layer L+1 up to its MoE router (attention half + mlp hyper-connection + gate),
    used to measure how the student's output of layer L moves the next layer's routing."""

    def __init__(self, g, il, cfg, runner, device="cuda"):
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
        W = {k: v for k, v in layer_weights(g, il, cfg).items()
             if not k.startswith(("mlp.experts.", "mlp.shared_expert"))}
        c = text_config(); c._attn_implementation = "eager"
        mod = Qwen4ExpTextDecoderLayer(c, il)
        mod.mlp.experts = nn.Identity(); mod.mlp.shared_expert = nn.Identity(); mod.mlp.shared_expert_gate = nn.Identity()
        miss, unexp = mod.load_state_dict({k: v.float() for k, v in W.items()}, strict=False)
        assert not unexp and all(k.startswith("ple.") for k in miss), (miss, unexp)
        assert mod.ple is None
        self.mod = mod.to(device).eval().requires_grad_(False)
        self.runner, self.H, self.k = runner, cfg.hidden_size, cfg.num_experts_per_tok

    def logits(self, x):
        m = self.mod
        h, hin, inj = m.attn_hyper_connection(x)
        if m.layer_type == "linear_attention":
            h = m.linear_attn(h, cache_params=None, attention_mask=None)
        else:
            pe, mask = self.runner.pe_mask(x)
            h, _ = m.self_attn(h, pe, attention_mask=mask)
        x2 = hin + (h.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
        h2, _, _ = m.mlp_hyper_connection(x2)
        return F.linear(h2.reshape(-1, self.H), m.mlp.gate.weight).float()


def router_kl(lt, ls):
    """KL(p_teacher || p_student) over the full expert softmax, mean over tokens."""
    return (F.softmax(lt, -1) * (F.log_softmax(lt, -1) - F.log_softmax(ls, -1))).sum(-1).mean()


def router_logit_mse(lt, ls):
    """||ls - lt||^2 / ||lt - mean_e(lt)||^2: relative error of the next router's logits,
    which is what decides the top-k ranking (softmax KL barely sees near-ties at the k boundary)."""
    return ((ls - lt) ** 2).sum() / ((lt - lt.mean(-1, keepdim=True)) ** 2).sum()


def topk_change(lt, ls, k):
    """1 - |top-k(teacher) ∩ top-k(student)| / k, mean over tokens."""
    a, b = lt.topk(k, -1).indices, ls.topk(k, -1).indices
    return float(1 - (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().sum(-1).mean() / k)


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
    ap.add_argument("--router-kl", type=float, default=0.0,
                    help="weight of the next-layer router term (teacher output vs student output); 0 = off")
    ap.add_argument("--router-loss", choices=["kl", "mse"], default="kl",
                    help="kl: softmax KL over all experts; mse: relative MSE of the router logits")
    ap.add_argument("--cos-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--train-dir", nargs="+", default=["/data/eval/q4x_act_train"],
                    help="one or more stream dumps, concatenated along the sequence axis")
    ap.add_argument("--valid-dir", default="/data/eval/q4x_act_valid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="layers whose experts_L{N}.npz already exists in --out are not retrained: their "
                         "saved values only advance the streams")
    ap.add_argument("--steps-schedule", default=None,
                    help="adaptive steps from the layer's PTQ delta relMSE, e.g. '0.15:300,0.22:600,inf:900' "
                         "(first threshold above the PTQ error wins); overrides --steps")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stop when the valid delta relMSE has not improved for this many steps "
                         "(checked every --eval-every); the best evaluated state is kept. 0 = off")
    ap.add_argument("--probe", type=int, default=0,
                    help="also report the stream error after passing the student output (and the canonical one) "
                         "through the next N unmodified teacher layers, at PTQ init and after training")
    ap.add_argument("--adam-bits", type=int, default=16, choices=[8, 16],
                    help="precision of the expert-latent Adam state (8 saves ~5 GB of VRAM)")
    ap.add_argument("--stream-dir", default="/data/eval/q4x_streams")
    ap.add_argument("--save-streams-at", type=int, nargs="*", default=[],
                    help="save the student/canonical streams entering these layers to --stream-dir")
    ap.add_argument("--load-streams", action="store_true",
                    help="start from streams saved at --first instead of --train-dir/--valid-dir")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    dev = "cuda"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = text_config(); width = cfg.hidden_size * cfg.hc_count
    runner = LayerRunner(cfg, dev)
    g = GGUFTensors(FLASH_GGUF)
    if a.load_streams:
        sd = Path(a.stream_dir) / f"L{a.first}"
        S_tr, S_va, C_tr, C_va = (torch.from_numpy(np.load(sd / f"{n}.npy")) for n in ("S_tr", "S_va", "C_tr", "C_va"))
    else:
        S_tr = torch.cat([load_stream(Path(d), width) for d in a.train_dir]); S_va = load_stream(Path(a.valid_dir), width)
        C_tr, C_va = S_tr.clone(), S_va.clone()
    print(f"streams: train {tuple(S_tr.shape)} valid {tuple(S_va.shape)}", flush=True)
    log = {"args": vars(a), "layers": {}}
    if a.resume and (out / "log.json").exists():
        log["layers"] = json.load(open(out / "log.json"))["layers"]

    def steps_for(ptq_err):
        if not a.steps_schedule:
            return a.steps
        for part in a.steps_schedule.split(","):
            thr, n = part.split(":")
            if ptq_err < float(thr):
                return int(n)
        return int(a.steps_schedule.split(",")[-1].split(":")[1])

    for L in range(a.first, a.last + 1):
        t0 = time.time(); ll = {}
        if L in a.save_streams_at:
            sd = Path(a.stream_dir) / f"L{L}"; sd.mkdir(parents=True, exist_ok=True)
            for n, t in (("S_tr", S_tr), ("S_va", S_va), ("C_tr", C_tr), ("C_va", C_va)):
                np.save(sd / f"{n}.npy", t.numpy())
            print(f"=== streams entering layer {L} saved to {sd}", flush=True)
        W = layer_weights(g, L, cfg)
        done = out / f"experts_L{L}.npz"
        if a.resume and done.exists() and str(L) in log["layers"]:
            teacher, _, _, _ = build_layer(L, W, device=dev)
            C_tr, C_va = run(teacher, runner, C_tr), run(teacher, runner, C_va)
            del teacher; torch.cuda.empty_cache()
            student, _, _, _ = build_layer(L, {k: v for k, v in W.items() if not k.startswith("mlp.experts.")}, device=dev)
            student.mlp.experts = TernaryExperts(W["mlp.experts.gate_up_proj"], W["mlp.experts.down_proj"], dev)
            del W
            student.mlp.experts.load_values(done)
            S_tr, S_va = run(student, runner, S_tr), run(student, runner, S_va)
            print(f"=== layer {L} resumed from {done.name}: stream vs canonical relMSE {rel(S_va, C_va):.4f} "
                  f"(logged {log['layers'][str(L)]['exit_stream_vsC']:.4f})  ({time.time()-t0:.0f}s)", flush=True)
            del student; gc.collect(); torch.cuda.empty_cache()
            continue
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

        nr = NextRouter(g, L + 1, cfg, runner, dev) if L + 1 < cfg.num_hidden_layers else None

        @torch.no_grad()
        def evaluate():
            Y = run(student, runner, S_va)
            dA = rel(Y.float() - S_va.float(), A_va.float() - S_va.float())
            out = {"delta_vsA": dA, "stream_vsC": rel(Y, Cx_va), "Y": Y}
            if nr is not None:                     # next-layer routing: student vs teacher(same input) / canonical
                kl = lm = chA = chC = 0.0; n = 0
                for i in range(0, Y.shape[0], 2):
                    ls = nr.logits(Y[i:i + 2].to(dev).float())
                    la = nr.logits(A_va[i:i + 2].to(dev).float()); lc = nr.logits(Cx_va[i:i + 2].to(dev).float())
                    kl += float(router_kl(la, ls)); lm += float(router_logit_mse(la, ls)); chA += topk_change(la, ls, nr.k); chC += topk_change(lc, ls, nr.k); n += 1
                out.update(next_kl_vsA=kl / n, next_logit_rel_vsA=lm / n, next_route_vsA=chA / n, next_route_vsC=chC / n)
            return out

        def fmt(e):
            s = f"delta vsA relMSE {e['delta_vsA']:.4f} | stream vsC relMSE {e['stream_vsC']:.4f}"
            if "next_kl_vsA" in e:
                s += (f" | next router KL vsA {e['next_kl_vsA']:.4f} logit relMSE {e['next_logit_rel_vsA']:.4f} route chg vsA {e['next_route_vsA']:.4f}"
                      f" vsC {e['next_route_vsC']:.4f}")
            return s
        probe_C = None

        @torch.no_grad()
        def probe(Y):
            """stream error after the next a.probe unmodified teacher layers (student vs canonical)."""
            nonlocal probe_C
            first = probe_C is None
            if first:
                probe_C = []
            xs, xc, res = Y, Cx_va, []
            for j in range(1, a.probe + 1):
                if L + j >= cfg.num_hidden_layers:
                    break
                tj, _, _, _ = build_layer(L + j, layer_weights(g, L + j, cfg), device=dev)
                xs = run(tj, runner, xs)
                if first:
                    xc = run(tj, runner, xc); probe_C.append(xc)
                del tj; torch.cuda.empty_cache()
                res.append(rel(xs, probe_C[j - 1]))
            return res

        e0 = evaluate(); ll["ptq"] = {k: v for k, v in e0.items() if k != "Y"}
        print(f"  PTQ init : {fmt(e0)}", flush=True)
        if a.probe:
            ll["ptq"]["probe"] = probe(e0["Y"])
            print(f"  PTQ probe: stream vsC after +1..+{a.probe} teacher layers {[round(v, 4) for v in ll['ptq']['probe']]}", flush=True)
        steps = steps_for(e0["delta_vsA"]); ll["steps"] = steps

        if steps > 0:
            ex.opt = FusedAdam({"gu": ex.gu_lat, "dn": ex.dn_lat}, lr=a.lr_w, state_bits=a.adam_bits)
            opt_s = torch.optim.AdamW([{"params": scl, "lr": a.lr_s}, {"params": nrm, "lr": a.lr_norm}], betas=(0.9, 0.99), weight_decay=0.0)
            sched = lambda t: 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps))  # noqa: E731
            gen = torch.Generator().manual_seed(L); tt = time.time()
            best, best_step, best_vals, traj = e0["delta_vsA"], 0, None, []
            print(f"    steps {steps}" + (f" (schedule, PTQ {e0['delta_vsA']:.4f})" if a.steps_schedule else "")
                  + (f", patience {a.patience}" if a.patience else ""), flush=True)
            for step in range(1, steps + 1):
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
                lr_kl = torch.zeros((), device=dev)
                if a.router_kl > 0 and nr is not None:
                    with torch.no_grad():
                        lt = nr.logits(yA)
                    lr_kl = (router_kl if a.router_loss == "kl" else router_logit_mse)(lt, nr.logits(y))
                    loss = loss + a.router_kl * lr_kl
                opt_s.zero_grad(set_to_none=True)
                loss.backward()                           # latents are updated inside backward
                ex.gu_lat.grad = None; ex.dn_lat.grad = None
                torch.nn.utils.clip_grad_norm_(scl + nrm, 1.0)
                opt_s.step()
                with torch.no_grad():
                    for s in scl: s.clamp_(min=1e-6)
                if step % 50 == 0:
                    print(f"    step {step:4d} loss {loss.item():.4f} A {lA.item():.4f} C {lC.item():.4f} "
                          f"R {lr_kl.item():.4f} ({time.time()-tt:.0f}s, peak {torch.cuda.max_memory_allocated()/2**30:.1f}G)", flush=True)
                if step % a.eval_every == 0 or step == steps:
                    ev = evaluate()
                    traj.append({"step": step, "delta_vsA": ev["delta_vsA"], "stream_vsC": ev["stream_vsC"]})
                    print(f"    [eval {step}] {fmt(ev)}", flush=True)
                    if ev["delta_vsA"] < best - 1e-4:
                        best, best_step = ev["delta_vsA"], step
                        if a.patience:
                            e = ex.export(); best_vals = {k: e[k].astype(np.float16) for k in ("gate", "up", "down")}; del e
                    elif a.patience and step - best_step >= a.patience:
                        print(f"    early stop at {step} (best {best:.4f} at {best_step})", flush=True)
                        break
                    del ev
            ex.opt = None
            # the last step's graph (loss -> ... -> TernaryMoE ctx) holds the optimizer state
            del loss, lA, lC, lcos, lr_kl, y, d, x, yA, yC, opt_s
            ll["trajectory"] = traj; ll["best_step"] = best_step; ll["stopped_at"] = step
            if best_vals is not None and best_step != step:
                ex.load_values(best_vals)                 # back to the best evaluated state
                print(f"    restored step {best_step}", flush=True)
            del best_vals
            e1 = evaluate(); ll["trained"] = {k: v for k, v in e1.items() if k != "Y"}
            if a.probe:
                ll["trained"]["probe"] = probe(e1["Y"])
                print(f"  trained probe: stream vsC after +1..+{a.probe} teacher layers {[round(v, 4) for v in ll['trained']['probe']]}", flush=True)
        else:
            e1 = e0
        # advance streams
        S_tr = run(student, runner, S_tr); S_va = e1["Y"]
        C_tr, C_va = Cx_tr, Cx_va
        exp = ex.export()
        save_experts(out / f"experts_L{L}.npz", {k: exp[k].astype(np.float16) for k in ("gate", "up", "down")})
        ll["exit_stream_vsC"] = rel(S_va, C_va)
        ll["zero_frac"] = float(sum((c == 0).float().mean() for c in exp["codes"].values()) / 3)
        print(f"  EXIT layer {L}: stream vs canonical relMSE {ll['exit_stream_vsC']:.4f}  zero {ll['zero_frac']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        log["layers"][str(L)] = ll
        json.dump(log, open(out / "log.json", "w"), indent=1)
        del student, ex, scl, nrm, nr, A_tr, A_va, e0, e1; gc.collect(); torch.cuda.empty_cache()
        print(f"  (layer end: allocated {torch.cuda.memory_allocated()/2**30:.2f}G)", flush=True)
        torch.cuda.reset_peak_memory_stats()
    print("Q4X PROGRESSIVE DONE", flush=True)


if __name__ == "__main__":
    main()
