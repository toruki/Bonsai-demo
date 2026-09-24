"""Replace selected tensors of an existing GGUF with Hadamard-folded ternary (PTQ1_0).

Everything else is copied byte-for-byte. The `prism.hadamard.*` contract is written for
exactly the replaced tensors, so the PrismML runtime rotates the activation only there.

    python gguf_inject_ternary.py --src model.gguf --dst out.gguf \
        --match 'ffn_(gate|up|down|gate_up)_exps' --block 128 --method mseopt

Weights can also come from a BF16 HuggingFace checkpoint instead of the GGUF itself
(`--hf-layer N`), which is what we want when the GGUF is already low-bit.
"""

from __future__ import annotations

import argparse
import os
import re
import time
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize, quant_shape_to_byte_shape  # noqa: E402

from bonsai_format import hadamard_matrix, ptq1_0_quantize  # noqa: E402
from ternary_store import load_experts  # noqa: E402


def make_signs(width: int, seed: int) -> np.ndarray:
    """Deterministic ±1 vector per input width (the manifest carries it, so any choice works)."""
    rng = np.random.default_rng((seed << 20) ^ width)
    return rng.choice(np.array([-1, 1], dtype=np.int8), width)


def ternarize(w: np.ndarray, group: int, method: str) -> np.ndarray:
    """w [.., group*k] f32 -> dequantized ternary (exactly representable in fp16*{-1,0,1})."""
    g = w.reshape(-1, group)
    a = np.abs(g)
    if method == "absmean":
        thr = 0.5 * a.mean(axis=1, keepdims=True)
        keep = a > thr
    else:                                    # exact per-group weight-MSE optimum
        srt = -np.sort(-a, axis=1)
        cs = np.cumsum(srt, axis=1)
        k = np.arange(1, group + 1, dtype=np.float32)
        kbest = np.argmax(cs * cs / k, axis=1)
        thr = srt[np.arange(srt.shape[0]), kbest][:, None]
        keep = a >= thr
    cnt = keep.sum(axis=1, keepdims=True)
    s = np.where(cnt > 0, (a * keep).sum(axis=1, keepdims=True) / np.maximum(cnt, 1), 0.0)
    s = s.astype(np.float16).astype(np.float32)
    return ((np.sign(g) * keep) * s).reshape(w.shape)


def fold(w: np.ndarray, block: int, signs: np.ndarray) -> np.ndarray:
    n_in = w.shape[-1]
    h = hadamard_matrix(block)
    return ((w * signs.astype(np.float32)).reshape(-1, block) @ h).reshape(w.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--match", default=r"ffn_(gate|up|down|gate_up)_exps\.weight$")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="only these blk.N (default: all)")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--method", default="mseopt", choices=["mseopt", "absmean"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-rotate", action="store_true", help="ternarize without the Hadamard fold/contract")
    ap.add_argument("--values-dir", default=None,
                    help="take already-folded ternary values from DIR/experts_L{N}.npz (keys gate/up/down) "
                         "instead of computing them; used to write back trained experts")
    ap.add_argument("--kv-only", action="store_true",
                    help="copy KVs and add the contract, but convert no tensor (for the KV shard of a split model)")
    ap.add_argument("--names", nargs="*", default=None, help="--kv-only: folded tensor names")
    ap.add_argument("--widths", type=int, nargs="*", default=None, help="--kv-only: their input widths")
    a = ap.parse_args()

    r = GGUFReader(a.src)
    arch_field = r.fields.get("general.architecture")
    arch = arch_field.contents() if arch_field else None   # non-first shards of a split carry no KV
    pat = re.compile(a.match)

    def targeted(name: str) -> bool:
        if not pat.search(name):
            return False
        if a.layers is not None:
            m = re.match(r"blk\.(\d+)\.", name)
            return m is not None and int(m.group(1)) in a.layers
        return True

    if a.kv_only:
        if not a.names or not a.widths:
            raise SystemExit("--kv-only needs --names and --widths")
        targets = []
        target_names = list(a.names)
        widths = sorted(set(a.widths))
    else:
        targets = [t for t in r.tensors if targeted(t.name)]
        if not targets:
            raise SystemExit(f"no tensor matched {a.match!r}")
        target_names = [t.name for t in targets]
        widths = sorted({int(t.shape[0]) for t in targets})
    for w in widths:
        if w % a.block or w % a.group:
            raise SystemExit(f"input width {w} not divisible by block {a.block} / group {a.group}")
    signs = {w: make_signs(w, a.seed) for w in widths}
    print(f"src={a.src}\narch={arch}  targets={len(target_names)}  widths={widths}  block={a.block} group={a.group} method={a.method}")

    w = GGUFWriter(a.dst, arch or "")
    if arch is None:
        # a split shard other than the first: reproduce its KV set exactly, no synthetic arch key
        w.kv_data[0].clear()
    for f in r.fields.values():
        if (arch is not None and f.name == gguf.Keys.General.ARCHITECTURE) or f.name.startswith("GGUF.") or f.name.startswith("prism.hadamard."):
            continue
        v = f.contents()
        if isinstance(v, (list, tuple)) and len(v) == 0:
            print(f"  skipping empty array KV {f.name}")
            continue
        w.add_key_value(f.name, v, f.types[0], sub_type=f.types[-1] if len(f.types) > 1 else None)
    if not a.no_rotate and arch is not None:
        w.add_uint32("prism.hadamard.version", 1)
        w.add_uint32("prism.hadamard.block_size", a.block)
        w.add_string("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard")
        w.add_string("prism.hadamard.axis", "input-last-dimension")
        w.add_string("prism.hadamard.sign_mode", "explicit")
        w.add_array("prism.hadamard.weight_names", target_names)
        w.add_array("prism.hadamard.sign_widths", [int(x) for x in widths])
        w.add_array("prism.hadamard.sign_values", [int(v) for wd in widths for v in signs[wd]])

    # two passes: sizes first (cheap), then compute + write one tensor at a time so the
    # working set never exceeds a single tensor (a whole-model run otherwise needs ~30 GB)
    UNQ = (GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16)
    _cache = {}

    def trained_values(layer: int) -> dict:
        """gate/up/down values of one layer from --values-dir (one layer kept at a time)."""
        if layer not in _cache:
            _cache.clear()
            _cache[layer] = load_experts(Path(a.values_dir) / f"experts_L{layer}.npz")
        return _cache[layer]

    def convert(t):
        """Fold + ternarize + pack, in chunks along the outermost axis.

        A whole expert tensor is 839 M params; materialising it as f32 and running the
        sort inside `ternarize` peaks around 20 GB, which is what kept killing the process
        on this machine. Chunking keeps the working set near 1 GB and is exact, because
        both the fold (blockwise on the last axis) and the group-128 quantiser act
        row-wise.
        """
        dims = [int(x) for x in t.shape]                  # gguf order, dims[0] = input axis
        logical = tuple(dims[::-1])
        if a.values_dir:
            m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", t.name)
            if not m:
                raise SystemExit(f"--values-dir has nothing for {t.name}")
            y = trained_values(int(m.group(1)))[m.group(2)].astype(np.float32)
            assert y.shape == logical, (t.name, y.shape, logical)
            parts = [ptq1_0_quantize(y[i:i + 16]) for i in range(0, y.shape[0], 16)]
            packed = np.concatenate(parts).reshape(quant_shape_to_byte_shape(logical, GGMLQuantizationType.PTQ1_0))
            return packed, float((y == 0).mean()), float("nan")
        inner = int(np.prod(logical[1:])) if len(logical) > 1 else 1
        step = max(1, (1 << 28) // max(inner * 4, 1))      # ~256 MB of f32 per chunk
        parts, err, ref, zeros, count = [], 0.0, 0.0, 0, 0
        for s0 in range(0, logical[0], step):
            raw = t.data[s0:s0 + step]
            x = (raw.astype(np.float32) if t.tensor_type in UNQ else dequantize(raw, t.tensor_type))
            x = x.reshape((-1,) + logical[1:])
            xf = x if a.no_rotate else fold(x, a.block, signs[dims[0]])
            del x
            y = ternarize(xf, a.group, a.method)
            err += float(((y - xf) ** 2).sum()); ref += float((xf ** 2).sum())
            zeros += int((y == 0).sum()); count += y.size
            del xf
            parts.append(ptq1_0_quantize(y))
            del y
        packed = np.concatenate(parts).reshape(quant_shape_to_byte_shape(logical, GGMLQuantizationType.PTQ1_0))
        return packed, zeros / count, err / ref

    for t in r.tensors:
        if a.kv_only or not targeted(t.name):
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            continue
        logical = tuple(int(x) for x in t.shape)[::-1]
        bshape = quant_shape_to_byte_shape(logical, GGMLQuantizationType.PTQ1_0)
        w.add_tensor_info(t.name, bshape, np.dtype(np.uint8), int(np.prod(bshape)), GGMLQuantizationType.PTQ1_0)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    total = sum(t.n_bytes for t in r.tensors)
    done = 0
    last_sync = 0
    for i, t in enumerate(r.tensors):
        if not a.kv_only and targeted(t.name):
            packed, zero, rel = convert(t)
            w.write_tensor_data(packed, tensor_endianess=r.endianess)
            print(f"  {t.name:34s} {t.tensor_type.name:7s} {[int(x) for x in t.shape]} -> PTQ1_0  "
                  f"zero={zero:.3f} relMSE={rel:.4f}", flush=True)
            del packed
        else:
            w.write_tensor_data(t.data, tensor_endianess=r.endianess)
        done += t.n_bytes
        # WSL2's virtual SCSI path runs out of swiotlb bounce buffers when the dirty page
        # pool grows without bound, which kills the process mid-write. Flush and drop the
        # cache for what we already wrote every few GB to keep it small.
        if done - last_sync > 256_000_000:
            for fo in w.fout:
                fo.flush()
                os.fsync(fo.fileno())
                os.posix_fadvise(fo.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            time.sleep(0.15)   # let the virtual storage queue drain
            last_sync = done
        if i % 50 == 0 or i + 1 == len(r.tensors):
            print(f"  [{i+1}/{len(r.tensors)}] {done/1e9:.1f}/{total/1e9:.1f} GB", flush=True)
    w.close()
    print("wrote", a.dst)


if __name__ == "__main__":
    main()
