"""Qwen3.8-Flash-Next (qwen4exp) decoder layer in PyTorch, with weights taken from a GGUF.

GGUF -> HF inverse of what conversion/qwen4exp.py (and its Qwen3Next / Qwen3.5 parents) does:
  * `*norm.weight` got `+1` (Gemma-style zero-centred gammas) except `linear_attn.norm.weight`
  * `A_log` became `ssm_a = -exp(A_log)`; `dt_bias` became `ssm_dt.bias`
  * linear-attention V heads were reordered grouped [nk, rep, hd] -> tiled [rep, nk, hd]
    in in_proj_qkv (V rows), in_proj_z, in_proj_a/b, A_log, dt_bias, conv1d (V channels)
    and out_proj (input columns)
  * experts.gate_up_proj [E, 2I, H] was split into ffn_gate_exps / ffn_up_exps
The reference layer is transformers' Qwen4ExpTextDecoderLayer, run in fp32.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "llama.cpp" / "gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

UNQ = (GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16)
FLASH_GGUF = sorted(glob.glob("/data/models/qwen3.8-flash-next/hf-cache/hub/*/snapshots/*/UD-IQ3_XXS/*.gguf"))
HF_REPO = "Qwen/Qwen3.8-Flash-Next"


def text_config():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(HF_REPO)
    return cfg.text_config if hasattr(cfg, "text_config") else cfg


class GGUFTensors:
    """Name -> tensor lookup across the shards of a (possibly split) GGUF."""

    def __init__(self, paths):
        self.readers = [GGUFReader(p) for p in paths]
        self.by = {t.name: t for r in self.readers for t in r.tensors}

    def f32(self, name: str) -> np.ndarray:
        t = self.by[name]
        logical = tuple(int(x) for x in t.shape)[::-1]
        if t.tensor_type in UNQ:
            if t.tensor_type == GGMLQuantizationType.BF16:
                x = (t.data.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                x = t.data.astype(np.float32)
        else:
            x = dequantize(t.data, t.tensor_type)
        return np.ascontiguousarray(x.reshape(logical))


def untile(w: np.ndarray, nk: int, rep: int, hd: int, axis: int = 0) -> np.ndarray:
    """tiled [rep, nk, hd] -> grouped [nk, rep, hd] along `axis`."""
    w = np.moveaxis(w, axis, 0)
    rest = w.shape[1:]
    w = w.reshape(rep, nk, hd, *rest).transpose(1, 0, 2, *range(3, 3 + len(rest))).reshape(rep * nk * hd, *rest)
    return np.ascontiguousarray(np.moveaxis(w, 0, axis))


def layer_weights(g: GGUFTensors, il: int, cfg) -> dict[str, torch.Tensor]:
    p = f"blk.{il}."
    nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    hk, hv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    rep = nv // nk
    has = lambda n: (p + n) in g.by  # noqa: E731
    out: dict[str, np.ndarray] = {}

    # hyper-connections
    for side, hf in (("attn", "attn_hyper_connection"), ("ffn", "mlp_hyper_connection")):
        out[f"{hf}.hc_norm.weight"] = g.f32(p + f"hc_{side}_norm.weight") - 1.0
        out[f"{hf}.input_mix_weight_down.weight"] = g.f32(p + f"hc_{side}_down.weight")
        out[f"{hf}.input_mix_weight_up.weight"] = g.f32(p + f"hc_{side}_up.weight")
        out[f"{hf}.block_inject_weight.weight"] = g.f32(p + f"hc_{side}_inject.weight")

    # MoE
    gate, up = g.f32(p + "ffn_gate_exps.weight"), g.f32(p + "ffn_up_exps.weight")      # [E, I, H]
    out["mlp.experts.gate_up_proj"] = np.concatenate([gate, up], axis=1)
    out["mlp.experts.down_proj"] = g.f32(p + "ffn_down_exps.weight")                    # [E, H, I]
    out["mlp.gate.weight"] = g.f32(p + "ffn_gate_inp.weight")                           # [E, H]
    out["mlp.shared_expert.gate_proj.weight"] = g.f32(p + "ffn_gate_shexp.weight")
    out["mlp.shared_expert.up_proj.weight"] = g.f32(p + "ffn_up_shexp.weight")
    out["mlp.shared_expert.down_proj.weight"] = g.f32(p + "ffn_down_shexp.weight")
    out["mlp.shared_expert_gate.weight"] = g.f32(p + "ffn_gate_inp_shexp.weight").reshape(1, -1)

    if has("ssm_out.weight"):                                   # Gated DeltaNet layer
        qkv = g.f32(p + "attn_qkv.weight")
        q, k, v = qkv[: nk * hk], qkv[nk * hk: 2 * nk * hk], qkv[2 * nk * hk:]
        out["linear_attn.in_proj_qkv.weight"] = np.concatenate([q, k, untile(v, nk, rep, hv)], 0)
        out["linear_attn.in_proj_z.weight"] = untile(g.f32(p + "attn_gate.weight"), nk, rep, hv)
        out["linear_attn.in_proj_a.weight"] = untile(g.f32(p + "ssm_alpha.weight"), nk, rep, 1)
        out["linear_attn.in_proj_b.weight"] = untile(g.f32(p + "ssm_beta.weight"), nk, rep, 1)
        out["linear_attn.A_log"] = np.log(-untile(g.f32(p + "ssm_a").reshape(-1), nk, rep, 1))
        out["linear_attn.dt_bias"] = untile(g.f32(p + "ssm_dt.bias").reshape(-1), nk, rep, 1)
        conv = g.f32(p + "ssm_conv1d.weight")                                          # [C, K]
        qk = 2 * nk * hk
        out["linear_attn.conv1d.weight"] = np.concatenate([conv[:qk], untile(conv[qk:], nk, rep, hv)], 0)[:, None, :]
        out["linear_attn.norm.weight"] = g.f32(p + "ssm_norm.weight").reshape(-1)
        out["linear_attn.out_proj.weight"] = untile(g.f32(p + "ssm_out.weight"), nk, rep, hv, axis=1)
    else:                                                       # QSA full-attention layer
        out["self_attn.q_proj.weight"] = g.f32(p + "attn_q.weight")         # q and gate, per head
        out["self_attn.k_proj.weight"] = g.f32(p + "attn_k.weight")
        out["self_attn.v_proj.weight"] = g.f32(p + "attn_v.weight")
        out["self_attn.o_proj.weight"] = g.f32(p + "attn_output.weight")
        out["self_attn.q_norm.weight"] = g.f32(p + "attn_q_norm.weight").reshape(-1) - 1.0
        out["self_attn.k_norm.weight"] = g.f32(p + "attn_k_norm.weight").reshape(-1) - 1.0
        out["self_attn.indexer.index_qk_proj.weight"] = np.concatenate(
            [g.f32(p + "indexer.q_proj.weight"), g.f32(p + "indexer.k_proj.weight")], 0)
        out["self_attn.indexer.q_layernorm.weight"] = g.f32(p + "indexer.q_norm.weight").reshape(-1) - 1.0
        out["self_attn.indexer.k_layernorm.weight"] = g.f32(p + "indexer.k_norm.weight").reshape(-1) - 1.0

    return {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in out.items()}


def build_layer(il: int, weights: dict[str, torch.Tensor], device="cuda", dtype=torch.float32):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
    cfg = text_config()
    cfg._attn_implementation = "eager"
    mod = Qwen4ExpTextDecoderLayer(cfg, il)
    missing, unexpected = mod.load_state_dict({k: v.to(dtype) for k, v in weights.items()}, strict=False)
    return mod.to(device=device, dtype=dtype).eval(), cfg, missing, unexpected


class LayerRunner:
    """Calls a Qwen4ExpTextDecoderLayer with what the model-level forward would pass it."""

    def __init__(self, cfg, device="cuda"):
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding
        self.rot = Qwen4ExpTextRotaryEmbedding(cfg).to(device)
        self.cache = {}

    def __call__(self, mod, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        if mod.layer_type == "linear_attention":
            return mod(x, position_embeddings=None, attention_mask=None)
        key = (B, L, x.dtype, x.device)
        if key not in self.cache:
            pos = torch.arange(L, device=x.device)[None, None, :].expand(3, B, L)
            pe = self.rot(x, pos)
            mask = torch.full((L, L), float("-inf"), device=x.device, dtype=x.dtype).triu(1)[None, None].expand(B, 1, L, L)
            self.cache[key] = (pe, mask)
        pe, mask = self.cache[key]
        return mod(x, position_embeddings=pe, attention_mask=mask)
