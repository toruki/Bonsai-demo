"""Export a mixed checkpoint: base BF16 everywhere, except selected layers replaced by
ternary layers (trained student states, or plain PTQ) stored folded as F16 with a
hadamard_packing.json covering exactly those tensors. The fork converter + runtime then
apply the rotation only where needed, so llama-perplexity measures the end-to-end effect
of ternarizing just those layers.

    python export_mixed.py --layers 32 33 34 35 --states DIR --dst /data/models/mixed-L32-35
    python export_mixed.py --layers 32 33 34 35 --ptq mseopt --dst /data/models/mixed-L32-35-ptq
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ternary_layer import (BASE, GROUP, TERNARY_LINEARS, TernaryLinear, build_layer, load_layer_weights,  # noqa: E402
                           shipped_signs, ternarize_layer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--states", default=None, help="dir with student_state_L{k}.pt (block) or student_state.pt (single)")
    ap.add_argument("--ptq", default=None, choices=["mseopt", "absmean"])
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    assert (a.states is None) != (a.ptq is None), "give --states or --ptq"
    dst = Path(a.dst); dst.mkdir(parents=True, exist_ok=True)

    # per-layer replacement tensors (HF names -> tensors) and manifest entries
    repl: dict[str, torch.Tensor] = {}
    entries = []
    for k in a.layers:
        prefix = f"model.language_model.layers.{k}."
        mod, _ = build_layer(k, load_layer_weights(k), device="cpu")
        mod = ternarize_layer(mod, init=a.ptq or "mseopt")
        if a.states:
            p = Path(a.states) / f"student_state_L{k}.pt"
            if not p.exists():
                p = Path(a.states) / "student_state.pt"
            sd = torch.load(p)
            missing, unexpected = mod.load_state_dict(sd, strict=False)
            assert not unexpected, unexpected
            print(f"layer {k}: loaded {len(sd)} tensors from {p}")
        with torch.no_grad():
            for n in TERNARY_LINEARS:
                try:
                    tl = mod.get_submodule(n)
                except AttributeError:
                    continue
                if not isinstance(tl, TernaryLinear):
                    continue
                wq = tl.quantized_weight()                       # folded ternary * fp16 scale, exact in F16
                repl[prefix + n + ".weight"] = wq.to(torch.float16).contiguous()
                entries.append({"name": prefix + n + ".weight", "axis": -1, "role": "fold-before-matmul"})
            for n, p in mod.named_parameters():
                if "norm" in n:
                    repl[prefix + n] = p.detach().to(torch.bfloat16).contiguous()
        zero = sum((v == 0).float().sum().item() for kk, v in repl.items() if kk.startswith(prefix) and v.dtype == torch.float16)
        tot = sum(v.numel() for kk, v in repl.items() if kk.startswith(prefix) and v.dtype == torch.float16)
        print(f"layer {k}: {tot/1e6:.0f}M ternary params, zero frac {zero/tot:.4f}")

    # shards: rewrite the ones that hold replaced tensors, symlink the rest
    index = json.load(open(BASE / "model.safetensors.index.json"))["weight_map"]
    touched = {index[n] for n in repl}
    for shard in sorted(set(index.values())):
        out = dst / shard
        if out.exists() or out.is_symlink():
            out.unlink()
        if shard in touched:
            t = load_file(str(BASE / shard))
            for n in list(t):
                if n in repl:
                    t[n] = repl[n]
            save_file(t, str(out), metadata={"format": "pt"})
            print(f"rewrote {shard}")
        else:
            os.symlink(BASE / shard, out)
    for f in BASE.iterdir():
        if f.suffix in (".json", ".txt", ".jinja") and f.name != "hadamard_packing.json":
            shutil.copy(f, dst / f.name)
    signs = shipped_signs()
    manifest = {"schema_version": 1, "kind": "hadamard-weight-fold", "status": "requires-matching-runtime",
                "producer": f"export_mixed.py layers={a.layers} states={a.states} ptq={a.ptq}",
                "transform": {"name": "normalized-signed-sylvester-walsh-hadamard", "block_size": 1024, "sign_mode": "explicit"},
                "signs": {str(w): [int(x) for x in v.tolist()] for w, v in signs.items()},
                "tensors": entries}
    json.dump(manifest, open(dst / "hadamard_packing.json", "w"))
    print(f"wrote {dst}: {len(entries)} folded tensors")


if __name__ == "__main__":
    main()
