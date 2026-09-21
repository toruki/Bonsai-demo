"""Ternary-aware Qwen3.5 decoder layer for output-reconstruction training.

Student = the stock transformers Qwen3_5DecoderLayer with every big nn.Linear swapped
for `TernaryLinear`, which keeps a latent full-precision weight and, on each forward,

    W_latent  ->  (W * s) @ H_1024  (shipped sign vector, runtime fold convention)
              ->  per-group-128 ternary with a learnable scale, STE on the round
              ->  y = (H (s ⊙ x)) @ Wq^T           (exactly what the PTQ1_0 runtime computes)

so the ternary *codes* are free to move during training (the latent weight crosses
the round boundaries), the group scales are trained, and RMSNorm weights are trained.
Everything else (in_proj_a/b, conv1d, A_log, dt_bias) is frozen by default.

Teacher = the same layer class with BF16 weights, unmodified.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "llama.cpp" / "gguf-py"))
from bonsai_format import hadamard_matrix, ptq1_0_dequantize, PTQ1_0_BLOCK_BYTES  # noqa: E402

BLOCK = 1024
GROUP = 128
BASE = Path("/data/models/qwen3.8-27b-bf16")
SHIPPED_GGUF = HERE.parents[2] / "models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
SHIPPED_PTQ = Path("/data/models/gguf/shipped-bonsai2-27b-PTQ1_0.gguf")

# HF linear names inside one decoder layer that Bonsai 2 ternarizes
TERNARY_LINEARS = (
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
# HF name -> GGUF name (per layer)
GGUF_NAME = {
    "linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate", "linear_attn.out_proj": "ssm_out",
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
}


# ------------------------------------------------------------------ quantizer

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def ternary_quant(wf: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """wf [rows, n_in] folded latent; scale [rows*n_in/GROUP] -> dequantized ternary weight.

    Value: q = clamp(round(wf/s), -1, 1) * s.
    Gradient (LSQ-style straight-through): d q / d wf = 1 inside the representable range
    (|wf/s| <= 1.5), 0 outside; d q / d s = t - wf/s inside, t outside.
    """
    g = wf.reshape(-1, GROUP)
    s = scale.reshape(-1, 1)
    u = g / s
    t = torch.clamp(torch.round(u), -1.0, 1.0)
    inside = (u.abs() <= 1.5).to(u.dtype)
    q = (t.detach() + (u - u.detach()) * inside) * s
    return q.reshape(wf.shape)


def ternary_codes(wf: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        u = (wf.reshape(-1, GROUP) / scale.reshape(-1, 1))
        return torch.clamp(torch.round(u), -1, 1).to(torch.int8).reshape(wf.shape)


class TernaryLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, signs: torch.Tensor, init: str = "mseopt"):
        """weight [out, in] (full precision), signs [in] in {-1,+1}."""
        super().__init__()
        n_out, n_in = weight.shape
        assert n_in % BLOCK == 0 and n_in % GROUP == 0
        self.n_out, self.n_in = n_out, n_in
        self.register_buffer("signs", signs.to(torch.float32))
        self.register_buffer("H", torch.from_numpy(hadamard_matrix(BLOCK)))
        # latent weight lives in the *primal* basis (what a checkpoint would store);
        # folding happens in forward so the rotation stays exactly the runtime's
        self.weight = nn.Parameter(weight.to(torch.float32).clone())
        with torch.no_grad():
            wf = self.fold(self.weight)
            s0 = self._init_scale(wf, init)
        self.scale = nn.Parameter(s0)
        # activation rotation reused for every token: R x = H (s ⊙ x)
        self.frozen_codes = None   # set to int8 codes to train scale/norm only

    @classmethod
    def from_folded(cls, wf: torch.Tensor, signs: torch.Tensor) -> "TernaryLinear":
        """Build from an already-folded ternary weight (e.g. the shipped GGUF): the latent
        is the unfolded weight, and the amax scale reproduces the codes exactly."""
        H = torch.from_numpy(hadamard_matrix(BLOCK)).to(wf.dtype)
        w = ((wf.reshape(-1, BLOCK) @ H).reshape(wf.shape)) * signs.to(wf.dtype)
        return cls(w, signs, init="amax")

    def fold(self, w: torch.Tensor) -> torch.Tensor:
        return ((w * self.signs).reshape(-1, BLOCK) @ self.H).reshape(self.n_out, self.n_in)

    @staticmethod
    def _init_scale(wf: torch.Tensor, init: str) -> torch.Tensor:
        g = wf.reshape(-1, GROUP)
        a = g.abs()
        if init == "absmean":
            thr = 0.5 * a.mean(dim=1, keepdim=True)
            keep = a > thr
            s = (a * keep).sum(1) / keep.sum(1).clamp(min=1)
        elif init == "mseopt":
            srt, _ = torch.sort(a, dim=1, descending=True)
            cs = torch.cumsum(srt, dim=1)
            k = torch.arange(1, GROUP + 1, dtype=srt.dtype, device=srt.device)
            kbest = (cs * cs / k).argmax(dim=1)
            thr = srt.gather(1, kbest[:, None])
            keep = a >= thr
            s = (a * keep).sum(1) / keep.sum(1).clamp(min=1)
        elif init == "amax":            # exact for inputs that are already ternary*scale
            s = a.max(dim=1).values
        else:
            raise ValueError(init)
        # round-to-nearest at scale s has threshold s/2; the support-mean scale from a
        # top-k support is not the same operating point, so re-derive the RTN-consistent
        # scale: the one whose s/2 threshold reproduces `keep` as closely as possible is
        # simply s itself when |w| in support >= s/2 -- true for both rules above.
        return s.clamp(min=1e-8)

    def quantized_weight(self) -> torch.Tensor:
        wf = self.fold(self.weight)
        if self.frozen_codes is not None:
            return (self.frozen_codes.to(wf.dtype).reshape(-1, GROUP) * self.scale.reshape(-1, 1)).reshape(wf.shape)
        return ternary_quant(wf, self.scale)

    def codes(self) -> torch.Tensor:
        if self.frozen_codes is not None:
            return self.frozen_codes
        return ternary_codes(self.fold(self.weight.detach()), self.scale.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        xr = ((x.reshape(-1, self.n_in) * self.signs).reshape(-1, BLOCK) @ self.H).reshape(-1, self.n_in)
        y = xr @ self.quantized_weight().t()
        return y.reshape(*shp[:-1], self.n_out).to(x.dtype)


# ----------------------------------------------------------------- loading

def load_config():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(BASE))
    return cfg.text_config if hasattr(cfg, "text_config") else cfg


def load_layer_weights(layer: int, src: Path = BASE) -> dict[str, torch.Tensor]:
    """All HF tensors of `model.language_model.layers.{layer}.` from the safetensors shards."""
    from safetensors import safe_open
    index = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
    prefix = f"model.language_model.layers.{layer}."
    out = {}
    by_file: dict[str, list[str]] = {}
    for k, f in index.items():
        if k.startswith(prefix):
            by_file.setdefault(f, []).append(k)
    for f, keys in by_file.items():
        with safe_open(str(src / f), "pt") as sf:
            for k in keys:
                out[k[len(prefix):]] = sf.get_tensor(k)
    return out


def build_layer(layer: int, weights: dict[str, torch.Tensor], device="cuda", dtype=torch.float32):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    cfg = load_config()
    mod = Qwen3_5DecoderLayer(cfg, layer)
    missing, unexpected = mod.load_state_dict({k: v.to(dtype) for k, v in weights.items()}, strict=False)
    assert not [m for m in missing if "inv_freq" not in m], missing
    assert not unexpected, unexpected
    return mod.to(device=device, dtype=dtype), cfg


def shipped_signs() -> dict[int, torch.Tensor]:
    from gguf import GGUFReader
    r = GGUFReader(str(SHIPPED_GGUF))
    widths = r.fields["prism.hadamard.sign_widths"].contents()
    vals = np.asarray(r.fields["prism.hadamard.sign_values"].contents(), dtype=np.float32)
    out, off = {}, 0
    for w in widths:
        out[int(w)] = torch.from_numpy(vals[off:off + w].copy())
        off += w
    return out


def ternarize_layer(mod: nn.Module, init: str = "mseopt") -> nn.Module:
    """Swap the Bonsai-ternarized linears of a decoder layer for TernaryLinear."""
    signs = shipped_signs()
    for name in TERNARY_LINEARS:
        parent_name, _, child = name.rpartition(".")
        try:
            parent = mod.get_submodule(parent_name) if parent_name else mod
        except AttributeError:
            continue                       # e.g. no self_attn on a linear-attention layer
        if not hasattr(parent, child):
            continue
        lin: nn.Linear = getattr(parent, child)
        tl = TernaryLinear(lin.weight.data, signs[lin.in_features], init=init).to(lin.weight.device)
        setattr(parent, child, tl)
    return mod


def shipped_layer_weights(layer: int, gguf_path: Path = SHIPPED_PTQ) -> dict[str, torch.Tensor]:
    """Dequantized shipped Bonsai 2 tensors for one layer, mapped to HF names.

    Ternary weights come back *folded* (rotated basis). RMSNorm weights are returned in
    HF convention (w - 1). ssm_out is in the shipped (grouped-V) order == HF order.
    in_proj_qkv / in_proj_z rows are in the converter's tiled-V order and are mapped
    back to HF grouped order here.
    """
    from gguf import GGUFReader
    r = GGUFReader(str(gguf_path))
    by = {t.name: t for t in r.tensors}
    cfg = load_config()
    nk, nv, hd = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_value_head_dim
    hk = cfg.linear_key_head_dim
    rep = nv // nk

    def deq(t):
        if t.tensor_type.name == "PTQ1_0":
            ne0 = int(t.shape[0]); nb = ne0 // 128
            raw = t.data.view(np.uint8).reshape(-1, PTQ1_0_BLOCK_BYTES)
            return torch.from_numpy(ptq1_0_dequantize(raw).reshape(-1, ne0).copy())
        shape = [int(x) for x in t.shape[::-1]]
        if t.tensor_type.name == "BF16":
            u = np.frombuffer(t.data.tobytes(), dtype="<u2").astype(np.uint32) << 16
            return torch.from_numpy(u.view(np.float32).reshape(shape).copy())
        return torch.from_numpy(np.array(t.data, dtype=np.float32).reshape(shape))

    def untile_rows(w, head_dim):   # tiled [rep, nk, hd] rows -> grouped [nk, rep, hd]
        return w.reshape(rep, nk, head_dim, -1).permute(1, 0, 2, 3).reshape(-1, w.shape[-1])

    p = f"blk.{layer}."
    out = {}
    inv = {v: k for k, v in GGUF_NAME.items()}
    for g, hf in inv.items():
        if p + g + ".weight" in by:
            w = deq(by[p + g + ".weight"])
            if hf == "linear_attn.in_proj_qkv":
                q, k, v = w[: nk * hk], w[nk * hk: 2 * nk * hk], w[2 * nk * hk:]
                w = torch.cat([q, k, untile_rows(v, hd)], 0)
            elif hf == "linear_attn.in_proj_z":
                w = untile_rows(w, hd)
            out[hf + ".weight"] = w
    if p + "ssm_alpha.weight" in by:
        def untile_heads(v):        # tiled [rep, nk] -> grouped [nk, rep] along dim 0
            return v.reshape(rep, nk, *v.shape[1:]).transpose(0, 1).reshape(v.shape)
        out["linear_attn.in_proj_a.weight"] = untile_heads(deq(by[p + "ssm_alpha.weight"]))
        out["linear_attn.in_proj_b.weight"] = untile_heads(deq(by[p + "ssm_beta.weight"]))
        out["linear_attn.A_log"] = torch.log(-untile_heads(deq(by[p + "ssm_a"]).reshape(-1)))
        out["linear_attn.dt_bias"] = untile_heads(deq(by[p + "ssm_dt.bias"]).reshape(-1))
        conv = deq(by[p + "ssm_conv1d.weight"]).reshape(-1, 4 if by[p + "ssm_conv1d.weight"].shape[0] == 4 else int(by[p + "ssm_conv1d.weight"].shape[0]))
        qk = 2 * nk * hk
        cv = conv[qk:].reshape(rep, nk, hd, -1).permute(1, 0, 2, 3).reshape(-1, conv.shape[-1])
        out["linear_attn.conv1d.weight"] = torch.cat([conv[:qk], cv], 0)[:, None, :]
    norm_map = {"attn_norm": "input_layernorm", "post_attention_norm": "post_attention_layernorm",
                "ssm_norm": "linear_attn.norm", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm"}
    for g, hf in norm_map.items():
        if p + g + ".weight" in by:
            w = deq(by[p + g + ".weight"]).reshape(-1)
            out[hf + ".weight"] = w - 1.0 if hf != "linear_attn.norm" else w
    return out


def shipped_codes(layer: int, name: str, gguf_path: Path = SHIPPED_PTQ) -> torch.Tensor:
    """int8 codes of a shipped tensor, in the same row order as the student's TernaryLinear."""
    w = shipped_layer_weights(layer, gguf_path)[name + ".weight"]
    return torch.sign(w).to(torch.int8)


# --------------------------------------------------------------- data / eval

def load_acts(d: Path, name: str) -> torch.Tensor:
    meta = json.load(open(d / "meta.json"))
    ne0, nt = meta["tensors"][name]
    n_seq, L = meta["n_sequences"], meta["seq_len"]
    x = np.fromfile(d / f"{name}.f32", dtype=np.float32).reshape(nt, ne0)
    return torch.from_numpy(x).reshape(n_seq, L, ne0)


def layer_forward(mod: nn.Module, x: torch.Tensor, cfg, rotary=None) -> torch.Tensor:
    """x [B, L, D] -> y [B, L, D] for a single decoder layer (fresh state, causal)."""
    B, L, _ = x.shape
    pos_emb = None
    if mod.block_type == "full_attention":
        if rotary is None:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
            rotary = Qwen3_5TextRotaryEmbedding(cfg).to(x.device)
        pos = torch.arange(L, device=x.device)[None, None, :].expand(3, B, L)
        pos_emb = rotary(x, pos)
        mask = torch.full((L, L), float("-inf"), device=x.device).triu(1)[None, None]
    else:
        mask = None
    return mod(x, position_embeddings=pos_emb, attention_mask=mask)


@torch.no_grad()
def recon_metrics(y_ref: torch.Tensor, y: torch.Tensor) -> dict:
    a = y_ref.reshape(-1, y_ref.shape[-1]).float(); b = y.reshape(-1, y.shape[-1]).float()
    cos = F.cosine_similarity(a, b, dim=1)
    return {"cos_mean": cos.mean().item(), "cos_min": cos.min().item(),
            "rel_mse": ((a - b) ** 2).sum().item() / (a ** 2).sum().item()}
