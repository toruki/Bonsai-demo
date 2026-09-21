"""Build a "PTQ-Bonsai" checkpoint from the base Qwen3.8-27B.

Applies, purely post-training, the transform that docs/bonsai_reverse_analysis.md
identified from the shipped Bonsai 2 27B:

    W_folded = (W * s) @ H_1024       (shipped sign vectors, runtime convention)
    ternary per group of 128 along the input axis, one fp16 scale per group

with one of two scale/threshold rules:

    absmean : threshold = 0.5 * mean|W'_g|, scale = mean|W'| over the support
              (the operating point Bonsai 2 sits at; reproduces ~90% of its codes)
    mseopt  : weight-MSE optimum (method D of ternary_quant.py)

The output is an HF-layout directory that the fork's convert_hf_to_gguf.py accepts
as-is: same shard files/tensor membership as the source, folded+ternarized tensors
stored as F16 (t * fp16(scale) is exact in F16), every other tensor copied through,
plus hadamard_packing.json so the converter writes the prism.hadamard.* contract.

Nothing in this script touches the shipped Bonsai model or the runtime.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "llama.cpp" / "gguf-py"))
from bonsai_format import hadamard_matrix  # noqa: E402
from ternary_quant import method_d  # noqa: E402

BLOCK = 1024
GROUP = 128

# HF tensor name suffixes that Bonsai 2 folds before the matmul (Phase 1 §10)
FOLD_SUFFIXES = (
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.out_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
)
LM_HEAD = "lm_head.weight"
EMBED = "model.language_model.embed_tokens.weight"
LAYER_RE = re.compile(r"^model\.language_model\.layers\.\d+\.")


def role_of(name: str) -> str | None:
    if name == LM_HEAD:
        return "fold-before-matmul"
    if name == EMBED:
        return "inverse-after-lookup"
    if LAYER_RE.match(name) and name.endswith(FOLD_SUFFIXES):
        return "fold-before-matmul"
    return None


def shipped_signs(gguf_path: Path) -> dict[int, np.ndarray]:
    from gguf import GGUFReader
    r = GGUFReader(str(gguf_path))
    widths = r.fields["prism.hadamard.sign_widths"].contents()
    vals = np.asarray(r.fields["prism.hadamard.sign_values"].contents(), dtype=np.float32)
    out, off = {}, 0
    for w in widths:
        out[int(w)] = vals[off:off + w]
        off += w
    assert int(r.fields["prism.hadamard.block_size"].contents()) == BLOCK
    return out


def ternary_absmean(w: torch.Tensor) -> torch.Tensor:
    """threshold 0.5*mean|w|, support-mean scale, fp16 scale. w: [rows, n_in] f32."""
    g = w.reshape(-1, GROUP)
    a = g.abs()
    thr = 0.5 * a.mean(dim=1, keepdim=True)
    keep = a > thr
    cnt = keep.sum(dim=1, keepdim=True)
    s = torch.where(cnt > 0, (a * keep).sum(dim=1, keepdim=True) / cnt.clamp(min=1), torch.zeros_like(thr))
    s = s.to(torch.float16).to(torch.float32)
    t = torch.sign(g) * keep
    return (t * s).reshape(w.shape)


def ternary_mseopt(w: torch.Tensor) -> torch.Tensor:
    """Exact per-group weight-MSE optimum.

    For a fixed support size k the best support is the k largest |w| and the best
    scale is their mean, giving MSE_k = sum(w^2) - (sum of top-k |w|)^2 / k. So one
    sort + cumsum per group enumerates every k exactly; no grid, no local search.
    (Phase 3's method D is a local search for the same objective; this is its exact
    form and is never worse.)
    """
    g = w.reshape(-1, GROUP)
    a = g.abs()
    srt, _ = torch.sort(a, dim=1, descending=True)
    cs = torch.cumsum(srt, dim=1)
    k = torch.arange(1, GROUP + 1, dtype=torch.float32)
    gain = cs * cs / k                      # maximise (sum top-k)^2 / k
    kbest = gain.argmax(dim=1)              # 0-based -> support size kbest+1
    thr = srt.gather(1, kbest[:, None])     # smallest |w| inside the support
    keep = a >= thr
    cnt = keep.sum(dim=1, keepdim=True)
    s = ((a * keep).sum(dim=1, keepdim=True) / cnt.clamp(min=1)).to(torch.float16).to(torch.float32)
    t = torch.sign(g) * keep
    return (t * s).reshape(w.shape)


def fold(w: torch.Tensor, s: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    n_in = w.shape[-1]
    return ((w * s).reshape(-1, BLOCK) @ h).reshape(w.shape[0], n_in)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="base HF checkpoint dir")
    ap.add_argument("--dst", required=True, help="output HF-layout dir")
    ap.add_argument("--rule", choices=["absmean", "mseopt", "none"], required=True,
                    help="none = fold only (F16, no ternary): control for the fold/runtime path")
    ap.add_argument("--signs-from", default=str(HERE.parents[1] / "models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
    ap.add_argument("--only-shards", nargs="*", default=None, help="process only these shard files (testing)")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    signs = shipped_signs(Path(a.signs_from))
    h = torch.from_numpy(hadamard_matrix(BLOCK))
    quant = {"absmean": ternary_absmean, "mseopt": ternary_mseopt, "none": lambda x: x}[a.rule]

    index = json.load(open(src / "model.safetensors.index.json"))
    shards = sorted(set(index["weight_map"].values()))
    if a.only_shards:
        shards = [s for s in shards if s in set(a.only_shards)]

    folded: list[dict] = []
    stats = {"n_folded": 0, "params_folded": 0, "zero_frac_sum": 0.0}
    t0 = time.time()
    for si, shard in enumerate(shards):
        out_path = dst / shard
        if out_path.exists():
            print(f"[{si+1}/{len(shards)}] {shard}: exists, skipping")
            # still need manifest entries for it
            hdr = json.load(open(dst / f".{shard}.manifest.json"))
            folded.extend(hdr)
            continue
        tensors = load_file(str(src / shard))
        out: dict[str, torch.Tensor] = {}
        entries: list[dict] = []
        for name, w in tensors.items():
            role = role_of(name)
            if role is None or w.ndim != 2:
                out[name] = w
                continue
            n_in = w.shape[-1]
            if n_in % BLOCK or n_in not in signs:
                raise RuntimeError(f"{name}: input width {n_in} has no shipped sign vector / block mismatch")
            wf = fold(w.to(torch.float32), torch.from_numpy(signs[n_in]), h)
            wq = quant(wf)
            out[name] = wq.to(torch.float16)
            entries.append({"name": name, "axis": -1, "role": role})
            stats["n_folded"] += 1
            stats["params_folded"] += wq.numel()
            stats["zero_frac_sum"] += float((wq == 0).float().mean())
        save_file(out, str(out_path), metadata={"format": "pt"})
        json.dump(entries, open(dst / f".{shard}.manifest.json", "w"))
        folded.extend(entries)
        print(f"[{si+1}/{len(shards)}] {shard}: {len(entries)} folded tensors  "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

    # sidecar files the converter needs
    for f in src.iterdir():
        if f.suffix in (".json", ".txt", ".jinja") and f.name != "model.safetensors.index.json":
            shutil.copy(f, dst / f.name)
    shutil.copy(src / "model.safetensors.index.json", dst / "model.safetensors.index.json")

    manifest = {
        "schema_version": 1,
        "kind": "hadamard-weight-fold",
        "status": "requires-matching-runtime",
        "producer": f"tools/flash_next_ternary/make_ptq_bonsai.py rule={a.rule}",
        "transform": {
            "name": "normalized-signed-sylvester-walsh-hadamard",
            "block_size": BLOCK,
            "sign_mode": "explicit",
        },
        "signs": {str(w): [int(x) for x in v] for w, v in signs.items()},
        "tensors": folded,
    }
    json.dump(manifest, open(dst / "hadamard_packing.json", "w"))
    if stats["n_folded"]:
        print(f"done: {stats['n_folded']} tensors, {stats['params_folded']/1e9:.2f}B params ternarized, "
              f"mean zero frac {stats['zero_frac_sum']/stats['n_folded']:.4f}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
