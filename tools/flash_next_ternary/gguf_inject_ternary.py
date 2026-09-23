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
import re
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

    new_data: dict[str, np.ndarray] = {}
    for t in r.tensors:
        if a.kv_only or not targeted(t.name):
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            continue
        dims = [int(x) for x in t.shape]                  # gguf order, dims[0] = input axis
        logical = tuple(dims[::-1])
        x = (t.data.astype(np.float32) if t.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16)
             else dequantize(t.data, t.tensor_type)).reshape(logical)
        xf = x if a.no_rotate else fold(x, a.block, signs[dims[0]])
        y = ternarize(xf, a.group, a.method)
        packed = ptq1_0_quantize(y).reshape(quant_shape_to_byte_shape(logical, GGMLQuantizationType.PTQ1_0))
        new_data[t.name] = packed
        zero = float((y == 0).mean())
        rel = float(((y - xf) ** 2).sum() / (xf ** 2).sum())
        print(f"  {t.name:34s} {t.tensor_type.name:7s} {dims} -> PTQ1_0  zero={zero:.3f} relMSE={rel:.4f}", flush=True)
        del x, xf, y
        w.add_tensor_info(t.name, packed.shape, packed.dtype, packed.nbytes, GGMLQuantizationType.PTQ1_0)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    total = sum(t.n_bytes for t in r.tensors)
    done = 0
    for i, t in enumerate(r.tensors):
        w.write_tensor_data(new_data[t.name] if t.name in new_data else t.data, tensor_endianess=r.endianess)
        done += t.n_bytes
        if i % 25 == 0 or i + 1 == len(r.tensors):
            print(f"  writing {i+1}/{len(r.tensors)}  {done/1e9:.1f}/{total/1e9:.1f} GB", flush=True)
    w.close()
    print("wrote", a.dst)


if __name__ == "__main__":
    main()
