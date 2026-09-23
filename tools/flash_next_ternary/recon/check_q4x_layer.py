"""Numeric check: transformers Qwen4ExpTextDecoderLayer vs the ported llama.cpp qwen4exp runtime.

Both run layer L on the same input (llama.cpp's l_last-(L-1)) with the same weights
(the GGUF, layer L expanded to F32 so llama.cpp does fp32 matmuls there too).

    python check_q4x_layer.py --gguf flashnext-L2f32.gguf --dump /data/eval/q4x_dump_L2 --layer 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q4x_layer import GGUFTensors, LayerRunner, build_layer, layer_weights, text_config  # noqa: E402


def load_dump(d: Path, name: str, width: int) -> torch.Tensor:
    x = np.fromfile(d / f"{name}.f32", dtype=np.float32)
    return torch.from_numpy(x.reshape(-1, width))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    L = a.layer
    cfg = text_config()
    width = cfg.hidden_size * cfg.hc_count

    g = GGUFTensors([a.gguf])
    types = {n: g.by[n].tensor_type.name for n in g.by if n.startswith(f"blk.{L}.")}
    print("layer", L, "tensor types:", sorted(set(types.values())))
    W = layer_weights(g, L, cfg)
    mod, _, missing, unexpected = build_layer(L, W, device=a.device)
    assert not missing and not unexpected, (missing, unexpected)

    d = Path(a.dump)
    x = load_dump(d, f"l_last-{L-1}", width)
    y_ref = load_dump(d, f"l_last-{L}", width)
    print("tokens:", x.shape[0])
    with torch.no_grad():
        y = LayerRunner(cfg, a.device)(mod, x[None].to(a.device)).float().cpu()[0]

    def stats(ref, out, label):
        err = out - ref
        rel = float((err ** 2).sum() / (ref ** 2).sum())
        cos = torch.nn.functional.cosine_similarity(ref, out, dim=1)
        dref, dout = ref - x, out - x                         # the layer's own contribution
        drel = float(((dout - dref) ** 2).sum() / (dref ** 2).sum())
        dcos = torch.nn.functional.cosine_similarity(dref, dout, dim=1)
        print(f"{label:28s} stream relMSE {rel:.3e}  cos min {cos.min():.8f} | "
              f"delta relMSE {drel:.3e}  delta cos mean {dcos.mean():.7f} min {dcos.min():.6f}  "
              f"max|err| {err.abs().max():.3e}")
        return drel
    drel = stats(y_ref, y, f"layer {L} (l_last)")
    json.dump({"layer": L, "delta_rel_mse": drel}, open(d / "check.json", "w"))


if __name__ == "__main__":
    main()
