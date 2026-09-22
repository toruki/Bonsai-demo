"""Progressive block-wise ternary reconstruction over layers [first, last].

For block b (layers 4b..4b+3 by default):
    S_b : student residual stream entering the block  (S_0 = token embeddings)
    C_b : canonical BF16 residual stream entering the block
    Teacher A : BF16 layers of the block applied to S_b  -> A_int[j], A_exit
    Teacher B : canonical C_{b+1} = BF16 layers applied to C_b
    loss = relMSE_delta(y, A_exit) + anchor * relMSE_delta(y, C_exit)
         + (1 - cos(y, A_exit)) + aux * sum_j relMSE_delta(h_j, A_int[j])
    then S_{b+1} = student block(S_b), C_{b+1} = C_exit.

Variants: --steps 0 gives the PTQ-only progressive baseline; --anchor 0 --aux 0 is the
plain same-input block-exit reconstruction; the default is the dual-teacher objective.
Streams are kept as f16 on disk under --out so a run can be resumed block by block.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ternary_layer import (BASE, TERNARY_LINEARS, TernaryLinear, build_layer, load_config,  # noqa: E402
                           load_layer_weights, recon_metrics, ternarize_layer)

ACT_DIRS = ["/data/eval/act_train", "/data/eval/act_train2"]
VALID_DIR = "/data/eval/act_valid"


# ------------------------------------------------------------------ helpers

def tokens_of(d: str) -> torch.Tensor:
    m = json.load(open(Path(d) / "meta.json"))
    return torch.tensor(m["tokens"], dtype=torch.long).reshape(m["n_sequences"], m["seq_len"])


def embed(tokens: torch.Tensor) -> torch.Tensor:
    from safetensors import safe_open
    index = json.load(open(BASE / "model.safetensors.index.json"))["weight_map"]
    name = "model.language_model.embed_tokens.weight"
    with safe_open(str(BASE / index[name]), "pt") as f:
        E = f.get_tensor(name)
    return E[tokens].to(torch.float16)                        # [N, L, D] f16


class Rotary:
    def __init__(self, cfg):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
        self.rot = Qwen3_5TextRotaryEmbedding(cfg).cuda()
        self.cache = {}

    def __call__(self, x):
        B, L, _ = x.shape
        key = (B, L)
        if key not in self.cache:
            pos = torch.arange(L, device=x.device)[None, None, :].expand(3, B, L)
            self.cache[key] = self.rot(x, pos)
        return self.cache[key]


def fwd(mod, x, rotary):
    if mod.block_type == "full_attention":
        L = x.shape[1]
        mask = torch.full((L, L), float("-inf"), device=x.device).triu(1)[None, None]
        return mod(x, position_embeddings=rotary(x), attention_mask=mask)
    return mod(x, position_embeddings=None, attention_mask=None)


@torch.no_grad()
def run_layers(layers, x_all: torch.Tensor, rotary, bs=4, keep_intermediate=False):
    """x_all f16 [N, L, D] on CPU -> outputs f16 on CPU (exit, and optionally per-layer)."""
    outs = [[] for _ in layers] if keep_intermediate else None
    exit_ = []
    for i in range(0, x_all.shape[0], bs):
        h = x_all[i:i + bs].cuda().float()
        for j, m in enumerate(layers):
            h = fwd(m, h, rotary)
            if keep_intermediate:
                outs[j].append(h.to(torch.float16).cpu())
        exit_.append(h.to(torch.float16).cpu())
    exit_ = torch.cat(exit_)
    return (exit_, [torch.cat(o) for o in outs]) if keep_intermediate else exit_


def save_f16(t: torch.Tensor, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    t.contiguous().numpy().tofile(p)
    json.dump(list(t.shape), open(p.with_suffix(".json"), "w"))


def load_f16(p: Path) -> torch.Tensor:
    shape = json.load(open(p.with_suffix(".json")))
    return torch.from_numpy(np.fromfile(p, dtype=np.float16).reshape(shape))


def code_agree(layers, ref):
    vals = []
    for li, m in enumerate(layers):
        for n in TERNARY_LINEARS:
            try:
                tl = m.get_submodule(n)
            except AttributeError:
                continue
            if isinstance(tl, TernaryLinear) and (li, n) in ref:
                vals.append((tl.codes().cpu() == ref[(li, n)]).float().mean().item())
    return sum(vals) / len(vals) if vals else float("nan")


# -------------------------------------------------------------------- train

def train_block(layers, S, A_exit, C_exit, A_int, valid, cfg, rotary, a, log):
    """S/A_exit/C_exit/A_int[j]: f16 CPU tensors over train seqs; valid: dict of same for valid."""
    import torch.utils.checkpoint as ck
    USE_CKPT = True
    p_w, p_s, p_n = [], [], []
    for m in layers:
        for name, p in m.named_parameters():
            if isinstance(m.get_submodule(name.rpartition(".")[0]), TernaryLinear):
                (p_s if name.endswith("scale") else p_w).append(p)
            elif "norm" in name:
                p_n.append(p)
            else:
                p.requires_grad_(False)
    opt = torch.optim.AdamW([{"params": p_w, "lr": a.lr_w}, {"params": p_s, "lr": a.lr_s}, {"params": p_n, "lr": a.lr_norm}],
                            betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(t, a.steps) / a.steps)))

    def block_fwd(x):
        hs = []
        h = x
        for m in layers:
            h = ck.checkpoint(lambda inp, m=m: fwd(m, inp, rotary), h, use_reentrant=False) if torch.is_grad_enabled() else fwd(m, h, rotary)
            hs.append(h)
        return h, hs

    def rel(d, d_ref):
        return ((d - d_ref) ** 2).sum() / ((d_ref ** 2).sum() + 1e-12)

    @torch.no_grad()
    def evaluate():
        for m in layers: m.eval()
        ys = []
        for i in range(0, valid["S"].shape[0], 4):
            ys.append(block_fwd(valid["S"][i:i + 4].cuda().float())[0].float().cpu())
        y = torch.cat(ys); x = valid["S"].float()
        out = {"vsA": recon_metrics(valid["A_exit"].float() - x, y - x),
               "vsC": recon_metrics(valid["C_exit"].float() - valid["C_in"].float(), y - valid["C_in"].float()),
               "stream_vsC": recon_metrics(valid["C_exit"].float(), y)}
        for m in layers: m.train()
        return out

    n = S.shape[0]; g = torch.Generator().manual_seed(0); t0 = time.time(); hist = []
    for m in layers: m.train()
    for step in range(1, a.steps + 1):
        idx = torch.randperm(n, generator=g)[: a.bs]
        x = S[idx].cuda().float(); yA = A_exit[idx].cuda().float(); yC = C_exit[idx].cuda().float()
        y, hs = block_fwd(x)
        d = y - x
        lA = rel(d, yA - x)
        lC = rel(d, yC - x) if a.anchor > 0 else torch.zeros((), device=x.device)
        lcos = 1 - F.cosine_similarity(y.reshape(-1, y.shape[-1]), yA.reshape(-1, y.shape[-1]), dim=1).mean()
        laux = torch.zeros((), device=x.device)
        if a.aux > 0:
            for j in range(len(layers) - 1):
                tj = A_int[j][idx].cuda().float()
                laux = laux + rel(hs[j] - x, tj - x)
        loss = lA + a.anchor * lC + a.cos_weight * lcos + a.aux * laux
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(p_w + p_s + p_n, 1.0)
        opt.step(); sched.step()
        if step % 50 == 0:
            print(f"    step {step:5d} loss {loss.item():.4f} A {lA.item():.4f} C {lC.item():.4f} cos {lcos.item():.4f} aux {laux.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(); hist.append({"step": step, **ev})
            print(f"    [eval {step}] delta vsA cos {ev['vsA']['cos_mean']:.4f} relMSE {ev['vsA']['rel_mse']:.4f} | "
                  f"delta vsC cos {ev['vsC']['cos_mean']:.4f} relMSE {ev['vsC']['rel_mse']:.4f} | stream vsC cos {ev['stream_vsC']['cos_mean']:.5f} relMSE {ev['stream_vsC']['rel_mse']:.4f}", flush=True)
    log["train_history"] = hist
    for p in p_w + p_s + p_n: p.requires_grad_(False)
    del opt
    torch.cuda.empty_cache()


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=15)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr-w", type=float, default=2e-4)
    ap.add_argument("--lr-s", type=float, default=1e-4)
    ap.add_argument("--lr-norm", type=float, default=1e-4)
    ap.add_argument("--cos-weight", type=float, default=1.0)
    ap.add_argument("--anchor", type=float, default=0.2, help="weight of the canonical-BF16 target")
    ap.add_argument("--aux", type=float, default=0.1, help="weight of intermediate-layer targets (teacher A)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--init", default="mseopt")
    ap.add_argument("--sanity-dir", default=None, help="llama.cpp dump dir (valid) with l_out-K to check the canonical stream")
    ap.add_argument("--compact", action="store_true", help="save codes+scales+norms instead of full latent states")
    ap.add_argument("--keep-streams", action="store_true", help="keep every boundary's f16 streams on disk")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(); rotary = Rotary(cfg)
    log_path = out / "log.json"
    log = json.load(open(log_path)) if log_path.exists() else {"args": vars(a), "blocks": {}}

    # streams at the entry of `first`
    tok_tr = torch.cat([tokens_of(d) for d in ACT_DIRS]); tok_va = tokens_of(VALID_DIR)
    if a.first == 0:
        S_tr = embed(tok_tr); S_va = embed(tok_va); C_tr, C_va = S_tr.clone(), S_va.clone()
    else:
        S_tr, S_va = load_f16(out / f"S_{a.first}_train.f16"), load_f16(out / f"S_{a.first}_valid.f16")
        if (out / f"C_{a.first}_train.f16").exists():
            C_tr, C_va = load_f16(out / f"C_{a.first}_train.f16"), load_f16(out / f"C_{a.first}_valid.f16")
        else:
            # canonical stream was cleaned up: recompute it from the embeddings through BF16 layers 0..first-1
            print(f"recomputing canonical stream through BF16 layers 0..{a.first-1}", flush=True)
            C_tr, C_va = embed(tok_tr), embed(tok_va)
            for k0 in range(0, a.first, 8):
                teachers = [build_layer(k, load_layer_weights(k))[0].eval() for k in range(k0, min(k0 + 8, a.first))]
                C_tr = run_layers(teachers, C_tr, rotary); C_va = run_layers(teachers, C_va, rotary)
                del teachers; torch.cuda.empty_cache()
    print(f"streams: train {tuple(S_tr.shape)} valid {tuple(S_va.shape)}  (f16 in RAM)")

    for b0 in range(a.first, a.last + 1, a.block):
        b1 = min(b0 + a.block - 1, a.last)
        ks = list(range(b0, b1 + 1))
        print(f"\n=== block layers {b0}..{b1} ===", flush=True)
        blog = {}
        # teachers
        teachers = [build_layer(k, load_layer_weights(k))[0].eval() for k in ks]
        t0 = time.time()
        A_exit_tr, A_int_tr = run_layers(teachers, S_tr, rotary, keep_intermediate=True)
        A_exit_va, A_int_va = run_layers(teachers, S_va, rotary, keep_intermediate=True)
        C_exit_tr = run_layers(teachers, C_tr, rotary)
        C_exit_va = run_layers(teachers, C_va, rotary)
        print(f"  teachers done ({time.time()-t0:.0f}s)", flush=True)
        if a.sanity_dir:
            p = Path(a.sanity_dir) / f"l_out-{b1}.f32"
            if p.exists():
                meta = json.load(open(Path(a.sanity_dir) / "meta.json"))
                ref = torch.from_numpy(np.fromfile(p, dtype=np.float32).reshape(meta["n_sequences"], meta["seq_len"], -1))
                m = recon_metrics(ref, C_exit_va.float())
                print(f"  SANITY canonical stream vs llama.cpp l_out-{b1}: cos {m['cos_mean']:.6f} relMSE {m['rel_mse']:.2e}", flush=True)
                blog["sanity_vs_llamacpp"] = m
        del teachers; torch.cuda.empty_cache()

        # student
        layers = [ternarize_layer(build_layer(k, load_layer_weights(k))[0], init=a.init) for k in ks]
        valid = {"S": S_va, "A_exit": A_exit_va, "C_exit": C_exit_va, "C_in": C_va}
        m0 = {"stream_vsC": recon_metrics(C_exit_va.float(), run_layers(layers, S_va, rotary).float()),
              "vsA": None}
        print(f"  INIT stream vsC: cos {m0['stream_vsC']['cos_mean']:.5f} relMSE {m0['stream_vsC']['rel_mse']:.4f}", flush=True)
        blog["init"] = m0
        if a.steps > 0:
            train_block(layers, S_tr, A_exit_tr, C_exit_tr, A_int_tr, valid, cfg, rotary, a, blog)
        # advance streams
        for m in layers: m.eval()
        S_tr = run_layers(layers, S_tr, rotary); S_va = run_layers(layers, S_va, rotary)
        C_tr, C_va = C_exit_tr, C_exit_va
        fin = recon_metrics(C_va.float(), S_va.float())
        print(f"  EXIT layer {b1}: student stream vs canonical: cos {fin['cos_mean']:.5f} relMSE {fin['rel_mse']:.4f}", flush=True)
        blog["exit_stream_vsC"] = fin
        for k, m in zip(ks, layers):
            if a.compact:
                # ternary codes (int8, folded basis) + scales + norms: all an export needs, ~0.4 GB/layer
                sd = {}
                for n in TERNARY_LINEARS:
                    try:
                        tl = m.get_submodule(n)
                    except AttributeError:
                        continue
                    if isinstance(tl, TernaryLinear):
                        sd[n + ".codes"] = tl.codes().cpu(); sd[n + ".scale"] = tl.scale.detach().cpu()
                for kk, v in m.named_parameters():
                    if "norm" in kk:
                        sd[kk] = v.detach().cpu()
                torch.save(sd, out / f"student_compact_L{k}.pt")
            else:
                torch.save({kk: v.detach().cpu() for kk, v in m.state_dict().items() if not kk.endswith((".H", ".signs"))},
                           out / f"student_state_L{k}.pt")
        save_f16(S_tr, out / f"S_{b1+1}_train.f16"); save_f16(S_va, out / f"S_{b1+1}_valid.f16")
        save_f16(C_tr, out / f"C_{b1+1}_train.f16"); save_f16(C_va, out / f"C_{b1+1}_valid.f16")
        if not a.keep_streams:
            for f in out.glob(f"[SC]_{b0}_*"):
                f.unlink()
        log["blocks"][f"{b0}-{b1}"] = blog
        json.dump(log, open(log_path, "w"), indent=1)
        del layers, A_exit_tr, A_int_tr, A_exit_va, A_int_va; torch.cuda.empty_cache()
    print("PROGRESSIVE DONE", out)


if __name__ == "__main__":
    main()
