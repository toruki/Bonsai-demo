"""Rebuild trained student layers from saved states (full or compact) and regenerate streams."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ternary_layer import (BLOCK, GROUP, TERNARY_LINEARS, TernaryLinear, build_layer,  # noqa: E402
                           load_layer_weights, ternarize_layer)
from bonsai_format import hadamard_matrix  # noqa: E402


def load_student_layer(k: int, state_dir: Path, device="cuda", trainable_latent=True):
    """Layer k with TernaryLinear modules restored from student_state_L{k}.pt (full latent)
    or student_compact_L{k}.pt (codes+scales -> latent initialised at the ternary points)."""
    mod, cfg = build_layer(k, load_layer_weights(k), device=device)
    mod = ternarize_layer(mod, init="mseopt")
    full = state_dir / f"student_state_L{k}.pt"
    comp = state_dir / f"student_compact_L{k}.pt"
    if full.exists():
        sd = torch.load(full)
        missing, unexpected = mod.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
    elif comp.exists():
        sd = torch.load(comp)
        mod.load_state_dict({kk: v for kk, v in sd.items() if "norm" in kk}, strict=False)
        H = torch.from_numpy(hadamard_matrix(BLOCK)).to(device)
        with torch.no_grad():
            for n in TERNARY_LINEARS:
                try:
                    tl = mod.get_submodule(n)
                except AttributeError:
                    continue
                if not isinstance(tl, TernaryLinear):
                    continue
                codes = sd[n + ".codes"].to(device).float().reshape(-1, GROUP)
                scale = sd[n + ".scale"].to(device).float().reshape(-1, 1)
                wq = (codes * scale).reshape(tl.n_out, tl.n_in)                # folded ternary
                latent = ((wq.reshape(-1, BLOCK) @ H).reshape(tl.n_out, tl.n_in)) * tl.signs  # unfold
                tl.weight.copy_(latent); tl.scale.copy_(scale.reshape(-1))
                if not trainable_latent:
                    tl.frozen_codes = sd[n + ".codes"].to(device)
    else:
        raise FileNotFoundError(f"no state for layer {k} in {state_dir}")
    return mod.eval(), cfg
