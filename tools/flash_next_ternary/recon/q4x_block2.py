"""Two-layer block reconstruction of Flash-Next routed experts (ternary, H128).

Same parts as q4x_progressive.py, but layers are trained in pairs (L, L+1) on the output after BOTH:

    student:  S --ternary L--> y1 --ternary L+1--> y2      (routing recomputed by the student every step)
    teacher:  S --IQ3 L-->     A1 --IQ3 L+1-->     A2      (same input S)
    canonical C --IQ3 L--> Cx1 --IQ3 L+1--> Cx2

    loss = rel(y2 vs A2) + cos(y2, A2) + aux * rel(y1 vs A1) + anchor * rel(y2 vs Cx2)

where rel(y vs t) is the relative MSE of the residual update (y - S) against (t - S). Both layers'
latents and scales train together; the routing error of L+1 caused by L is inside the objective.
Writes experts_L{N}.npz per layer (same store as q4x_progressive.py), so eval_q4x_variant.sh and
--resume work unchanged.

    python q4x_block2.py --first 4 --last 11 --steps 300 --out DIR \
        --train-dir /data/eval/q4x_act_train .../act_train_256_1024 --adam-bits 8 --latent-dtype fp16 --max-vram 24
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from q4x_progressive import (FLASH_GGUF, LATENT_DTYPES, FusedAdam, GGUFTensors, LayerRunner, TernaryExperts,
                             build_layer, layer_weights, load_stream, rel, run, save_experts, text_config)


def build_student(g_w, L, dev, dtype):
    st, _, _, _ = build_layer(L, {k: v for k, v in g_w.items() if not k.startswith("mlp.experts.")}, device=dev,
                              drop_experts=True)
    st.mlp.experts = TernaryExperts(g_w["mlp.experts.gate_up_proj"], g_w["mlp.experts.down_proj"], dev,
                                    latent_dtype=dtype)
    for p in st.parameters():
        p.requires_grad_(False)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=4)
    ap.add_argument("--last", type=int, default=11)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr-w", type=float, default=1e-4)
    ap.add_argument("--lr-s", type=float, default=1e-4)
    ap.add_argument("--anchor", type=float, default=0.2)
    ap.add_argument("--aux", type=float, default=0.1, help="weight of the layer-L output term")
    ap.add_argument("--cos-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--train-dir", nargs="+", default=["/data/eval/q4x_act_train"])
    ap.add_argument("--valid-dir", default="/data/eval/q4x_act_valid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--adam-bits", type=int, default=8, choices=[8, 16])
    ap.add_argument("--latent-dtype", default="fp16", choices=list(LATENT_DTYPES))
    ap.add_argument("--max-vram", type=float, default=24)
    a = ap.parse_args()
    assert (a.last - a.first) % 2 == 1, "the range must hold whole pairs"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    dev = "cuda"
    if a.max_vram:
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_vram * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    ldt = LATENT_DTYPES[a.latent_dtype]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = text_config(); width = cfg.hidden_size * cfg.hc_count
    runner = LayerRunner(cfg, dev)
    g = GGUFTensors(FLASH_GGUF)
    S_tr = torch.cat([load_stream(Path(d), width) for d in a.train_dir]); S_va = load_stream(Path(a.valid_dir), width)
    C_tr, C_va = S_tr.clone(), S_va.clone()
    print(f"streams: train {tuple(S_tr.shape)} valid {tuple(S_va.shape)}", flush=True)
    log = {"args": vars(a), "layers": {}}
    if a.resume and (out / "log.json").exists():
        log["layers"] = json.load(open(out / "log.json"))["layers"]

    for L in range(a.first, a.last + 1, 2):
        t0 = time.time()
        W0, W1 = layer_weights(g, L, cfg), layer_weights(g, L + 1, cfg)
        done = [out / f"experts_L{L}.npz", out / f"experts_L{L + 1}.npz"]
        if a.resume and all(p.exists() for p in done) and str(L + 1) in log["layers"]:
            for Lx, W, npz in ((L, W0, done[0]), (L + 1, W1, done[1])):
                te, _, _, _ = build_layer(Lx, W, device=dev)
                C_tr, C_va = run(te, runner, C_tr), run(te, runner, C_va)
                del te; torch.cuda.empty_cache()
                st = build_student(W, Lx, dev, ldt); st.mlp.experts.load_values(npz)
                S_tr, S_va = run(st, runner, S_tr), run(st, runner, S_va)
                del st; gc.collect(); torch.cuda.empty_cache()
            print(f"=== block {L}-{L + 1} resumed: stream vs canonical {rel(S_va, C_va):.4f} "
                  f"(logged {log['layers'][str(L + 1)]['exit_stream_vsC']:.4f})  ({time.time()-t0:.0f}s)", flush=True)
            continue

        # teachers: A1 = T_L(S), A2 = T_{L+1}(A1); canonical Cx1 = T_L(C), Cx2 = T_{L+1}(Cx1)
        te, _, miss, unexp = build_layer(L, W0, device=dev); assert not miss and not unexp
        A1_tr, A1_va = run(te, runner, S_tr), run(te, runner, S_va)
        Cx1_tr, Cx1_va = run(te, runner, C_tr), run(te, runner, C_va)
        del te, C_tr; torch.cuda.empty_cache()
        te, _, miss, unexp = build_layer(L + 1, W1, device=dev); assert not miss and not unexp
        A2_tr, A2_va = run(te, runner, A1_tr), run(te, runner, A1_va)
        Cx2_tr, Cx2_va = run(te, runner, Cx1_tr), run(te, runner, Cx1_va)
        del te, Cx1_tr; torch.cuda.empty_cache()
        print(f"\n=== block {L}-{L + 1} ({cfg.layer_types[L]}, {cfg.layer_types[L + 1]}) teachers {time.time()-t0:.0f}s", flush=True)

        st0 = build_student(W0, L, dev, ldt); st1 = build_student(W1, L + 1, dev, ldt); del W0, W1
        ex0, ex1 = st0.mlp.experts, st1.mlp.experts
        scl = [ex0.gu_s, ex0.dn_s, ex1.gu_s, ex1.dn_s]
        for p in [ex0.gu_lat, ex0.dn_lat, ex1.gu_lat, ex1.dn_lat] + scl:
            p.requires_grad_(True)
        print(f"  (students built: allocated {torch.cuda.memory_allocated()/2**30:.1f}G, "
              f"peak so far {torch.cuda.max_memory_allocated()/2**30:.1f}G)", flush=True)
        torch.cuda.reset_peak_memory_stats()

        @torch.no_grad()
        def evaluate():
            Y1 = run(st0, runner, S_va); Y2 = run(st1, runner, Y1)
            return {"block_delta_vsA2": rel(Y2.float() - S_va.float(), A2_va.float() - S_va.float()),
                    "l1_delta_vsA1": rel(Y1.float() - S_va.float(), A1_va.float() - S_va.float()),
                    "exit1_vsC": rel(Y1, Cx1_va), "exit2_vsC": rel(Y2, Cx2_va), "Y1": Y1, "Y2": Y2}

        def fmt(e):
            return (f"block delta vsA2 {e['block_delta_vsA2']:.4f} | L{L} delta vsA1 {e['l1_delta_vsA1']:.4f} | "
                    f"stream vsC L{L} {e['exit1_vsC']:.4f} L{L + 1} {e['exit2_vsC']:.4f}")

        e0 = evaluate(); ll = {"ptq": {k: v for k, v in e0.items() if not k.startswith("Y")}}
        print(f"  PTQ init : {fmt(e0)}", flush=True)
        del e0

        for ex in (ex0, ex1):
            ex.opt = FusedAdam({"gu": ex.gu_lat, "dn": ex.dn_lat}, lr=a.lr_w, state_bits=a.adam_bits)
        opt_s = torch.optim.AdamW(scl, lr=a.lr_s, betas=(0.9, 0.99), weight_decay=0.0)
        sched = lambda t: 0.5 * (1 + math.cos(math.pi * min(t, a.steps) / a.steps))  # noqa: E731
        gen = torch.Generator().manual_seed(L); tt = time.time()
        for step in range(1, a.steps + 1):
            f = sched(step)
            for ex in (ex0, ex1):
                ex.opt.lr = a.lr_w * f; ex.opt.t = step
            for pg in opt_s.param_groups:
                pg["lr"] = a.lr_s * f
            ix = torch.randperm(S_tr.shape[0], generator=gen)[: a.bs]
            x = S_tr[ix].to(dev).float()
            a1, a2, c2 = (T[ix].to(dev).float() for T in (A1_tr, A2_tr, Cx2_tr))
            y1 = runner(st0, x).float()
            y2 = runner(st1, y1).float()

            def rd(y, t):
                return (((y - x) - (t - x)) ** 2).sum() / ((t - x) ** 2).sum()
            l_main, l_aux, l_c = rd(y2, a2), rd(y1, a1), rd(y2, c2)
            l_cos = 1 - F.cosine_similarity(y2.reshape(-1, width), a2.reshape(-1, width), dim=1).mean()
            loss = l_main + a.cos_weight * l_cos + a.aux * l_aux + a.anchor * l_c
            opt_s.zero_grad(set_to_none=True)
            loss.backward()                               # both layers' latents update inside backward
            for ex in (ex0, ex1):
                ex.gu_lat.grad = None; ex.dn_lat.grad = None
            torch.nn.utils.clip_grad_norm_(scl, 1.0)
            opt_s.step()
            with torch.no_grad():
                for s in scl:
                    s.clamp_(min=1e-6)
            if step % 50 == 0:
                print(f"    step {step:4d} loss {loss.item():.4f} main {l_main.item():.4f} aux {l_aux.item():.4f} "
                      f"C {l_c.item():.4f} ({time.time()-tt:.0f}s, peak {torch.cuda.max_memory_allocated()/2**30:.1f}G)",
                      flush=True)
            if step % a.eval_every == 0 and step != a.steps:
                ev = evaluate(); print(f"    [eval {step}] {fmt(ev)}", flush=True); del ev
        for ex in (ex0, ex1):
            ex.opt = None
        del loss, l_main, l_aux, l_c, l_cos, y1, y2, x, a1, a2, c2, opt_s
        e1 = evaluate(); ll["trained"] = {k: v for k, v in e1.items() if not k.startswith("Y")}
        print(f"  trained  : {fmt(e1)}", flush=True)

        # advance the streams through both trained layers
        del A1_tr, A2_tr; gc.collect()
        S_mid = run(st0, runner, S_tr); S_tr = run(st1, runner, S_mid); del S_mid
        S_va = e1["Y2"]; C_tr, C_va = Cx2_tr, Cx2_va
        for Lx, ex in ((L, ex0), (L + 1, ex1)):
            exp = ex.export()
            save_experts(out / f"experts_L{Lx}.npz", {k: exp[k] for k in ("gate", "up", "down")})
            log["layers"][str(Lx)] = {"block": [L, L + 1], "zero_frac": float(exp["zero_frac"]),
                                      "exit_stream_vsC": e1["exit1_vsC"] if Lx == L else e1["exit2_vsC"]}
            del exp
        log["layers"][str(L + 1)].update(ll)
        json.dump(log, open(out / "log.json", "w"), indent=1)
        print(f"  EXIT block {L}-{L + 1}: stream vs canonical L{L} {e1['exit1_vsC']:.4f} L{L + 1} {e1['exit2_vsC']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del st0, st1, ex0, ex1, scl, A1_va, A2_va, Cx1_va, e1; gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print("Q4X BLOCK2 DONE", flush=True)


if __name__ == "__main__":
    main()
