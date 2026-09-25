"""Checkpoint of a TernaryExperts training state: the latents themselves, not just code * scale.

experts_L{N}.npz keeps only the exported ternary values, so a restart from it puts every latent at
the centre of its code and the scale updates are the only thing that can move (see the progressive
pilot doc, section 14). This keeps what a real warm start needs:

    gu_lat, dn_lat   latents in their training dtype (fp16: 5.0 GB per Flash-Next layer)
    gu_s, dn_s       fp32 group scales
    (optional) the FusedAdam state, so a continued run does not restart the moments from zero

    save_latents(path, experts, opt=None)
    load_latents(path, experts, opt=None)    # shapes must match; the latent dtype follows `experts`
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_PARAMS = ("gu_lat", "dn_lat", "gu_s", "dn_s")


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy() if t.dtype != torch.bfloat16 else t.detach().float().cpu().numpy()


@torch.no_grad()
def save_latents(path, experts, opt=None) -> None:
    arrs = {k: _np(getattr(experts, k)) for k in _PARAMS}
    if opt is not None:
        arrs["opt_bits"] = np.array(opt.bits)
        arrs["opt_t"] = np.array(opt.t)
        names = ("m", "v") if opt.bits == 16 else ("mq", "vq", "ms", "vs")
        for n in names:
            for key, t in getattr(opt, n).items():
                arrs[f"opt_{n}_{key}"] = _np(t)
    path = Path(path)
    np.savez(path, **arrs)
    need = sum(a.nbytes for a in arrs.values())
    if path.stat().st_size < need:                                 # a full disk truncates silently
        raise RuntimeError(f"{path} truncated: {path.stat().st_size} < {need} bytes")


@torch.no_grad()
def load_latents(path, experts, opt=None) -> None:
    z = np.load(path)
    for k in _PARAMS:
        p = getattr(experts, k)
        src = torch.from_numpy(z[k])
        assert tuple(src.shape) == tuple(p.shape), (k, src.shape, p.shape)
        for i in range(0, p.shape[0], 32):                         # chunked: no second full copy on GPU
            p[i:i + 32].copy_(src[i:i + 32].to(p.device, p.dtype))
    if opt is not None and "opt_bits" in z.files:
        assert int(z["opt_bits"]) == opt.bits, "Adam state precision differs from the checkpoint"
        opt.t = int(z["opt_t"])
        names = ("m", "v") if opt.bits == 16 else ("mq", "vq", "ms", "vs")
        for n in names:
            for key, t in getattr(opt, n).items():
                src = torch.from_numpy(z[f"opt_{n}_{key}"])
                for i in range(0, t.shape[0], 32):
                    t[i:i + 32].copy_(src[i:i + 32].to(t.device, t.dtype))
