"""Is the ternary residual predictable from the layer's own MoE input?

For one layer, freezes the trained ternary experts (experts_L{N}.npz) and fits a ridge regression
from the MoE input h (after the attention half and the mlp hyper-connection, [tokens, H]) to the
routed-MoE output difference r = MoE_teacher(h) - MoE_student(h). Routing is identical for both (the
router is unchanged), so r is purely the expert quantisation residual. The fit uses train sequences
(the ridge lambda is chosen on a held-out part of train); the valid sequences are evaluation only.

Reports the fraction of ||r||^2 explained on train and valid, and the layer-output delta relMSE when
the fitted linear correction is added to the student's MoE output. If valid explained variance stays
near 0, no linear function of the input (which is what a dense correction path can learn most easily)
can cancel the residual.

    python residual_probe.py --layer 16 --npz /data/eval/q4x_l16_base/experts_L16.npz --n-train 256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from q4x_layer import FLASH_GGUF, GGUFTensors, LayerRunner, build_layer, layer_weights, text_config
from q4x_progressive import TernaryExperts, rel, run


@torch.no_grad()
def moe_input(mod, runner, x):
    h, hin, inj = mod.attn_hyper_connection(x)
    if mod.layer_type == "linear_attention":
        h = mod.linear_attn(h, cache_params=None, attention_mask=None)
    else:
        pe, mask = runner.pe_mask(x)
        h, _ = mod.self_attn(h, pe, attention_mask=mask)
    x2 = hin + (h.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
    h2, _, _ = mod.mlp_hyper_connection(x2)
    return h2.reshape(-1, h2.shape[-1])


class Corrected(torch.nn.Module):
    """student experts + h @ W (the fitted linear correction)."""

    def __init__(self, experts, W, b):
        super().__init__()
        self.experts, self.W, self.b = experts, W, b

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return self.experts(hidden_states, top_k_index, top_k_weights) + hidden_states @ self.W + self.b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--stream-dir", default="/data/eval/q4x_streams")
    ap.add_argument("--n-train", type=int, default=256, help="train sequences used for the fit")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[1e-2, 1e-1, 1, 10, 100, 1000])
    a = ap.parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = text_config(); H = cfg.hidden_size
    runner = LayerRunner(cfg, dev)
    g = GGUFTensors(FLASH_GGUF)
    sd = Path(a.stream_dir) / f"L{a.layer}"
    S_tr = torch.from_numpy(np.load(sd / "S_tr.npy"))[: a.n_train]; S_va = torch.from_numpy(np.load(sd / "S_va.npy"))
    W = layer_weights(g, a.layer, cfg)
    teacher, _, _, _ = build_layer(a.layer, W, device=dev)
    student, _, _, _ = build_layer(a.layer, {k: v for k, v in W.items() if not k.startswith("mlp.experts.")}, device=dev,
                                   drop_experts=True)
    student.mlp.experts = TernaryExperts(W["mlp.experts.gate_up_proj"], W["mlp.experts.down_proj"], dev)
    student.mlp.experts.load_values(a.npz)
    del W
    tex, sex, router = teacher.mlp.experts, student.mlp.experts, teacher.mlp.gate

    @torch.no_grad()
    def feats(S):
        """-> h [N, H], r [N, H] (teacher - student routed MoE output on the same h), in fp32 on CPU."""
        hs, rs = [], []
        for i in range(0, S.shape[0], 2):
            h = moe_input(teacher, runner, S[i:i + 2].to(dev).float())
            _, wts, idx = router(h)
            r = tex(h, idx, wts) - sex(h, idx, wts)
            hs.append(h.cpu()); rs.append(r.cpu())
        return torch.cat(hs), torch.cat(rs)

    h_tr, r_tr = feats(S_tr); h_va, r_va = feats(S_va)
    print(f"train tokens {h_tr.shape[0]}  valid tokens {h_va.shape[0]}", flush=True)
    # ridge: r ~ h W + b ; centre both
    n_fit = int(h_tr.shape[0] * 0.8)
    hm, rm = h_tr[:n_fit].mean(0), r_tr[:n_fit].mean(0)
    Xf, Yf = (h_tr[:n_fit] - hm).to(dev).double(), (r_tr[:n_fit] - rm).to(dev).double()
    Xh, Yh = (h_tr[n_fit:] - hm).to(dev).double(), (r_tr[n_fit:] - rm).to(dev).double()
    XtX, XtY = Xf.t() @ Xf, Xf.t() @ Yf
    eye = torch.eye(H, device=dev, dtype=torch.float64)
    best = None
    for lam in a.lambdas:
        Wr = torch.linalg.solve(XtX + lam * n_fit * eye, XtY)
        ev_fit = 1 - float(((Yf - Xf @ Wr) ** 2).sum() / (Yf ** 2).sum())
        ev_hold = 1 - float(((Yh - Xh @ Wr) ** 2).sum() / (Yh ** 2).sum())
        print(f"  lambda {lam:8g}: explained fit {ev_fit:.4f}  held-out train {ev_hold:.4f}", flush=True)
        if best is None or ev_hold > best[1]:
            best = (lam, ev_hold, Wr)
    lam, _, Wr = best
    Xv, Yv = (h_va - hm).to(dev).double(), (r_va - rm).to(dev).double()
    ev_va = 1 - float(((Yv - Xv @ Wr) ** 2).sum() / (Yv ** 2).sum())
    print(f"chosen lambda {lam:g}: explained variance on VALID = {ev_va:.4f}", flush=True)

    # put the correction back into the layer and measure delta vsA on valid
    Wf, bf = Wr.float(), (rm.to(dev) - hm.to(dev) @ Wr.float())
    A_va = run(teacher, runner, S_va)
    Y0 = run(student, runner, S_va)
    student.mlp.experts = Corrected(sex, Wf, bf)
    Y1 = run(student, runner, S_va)
    d0 = rel(Y0.float() - S_va.float(), A_va.float() - S_va.float())
    d1 = rel(Y1.float() - S_va.float(), A_va.float() - S_va.float())
    print(f"layer {a.layer} valid delta vsA: student {d0:.4f} -> with linear correction {d1:.4f}", flush=True)


if __name__ == "__main__":
    main()
