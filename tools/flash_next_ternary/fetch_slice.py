"""Pull a single tensor (or a slice of one) out of a HuggingFace safetensors repo.

The Flash-Next BF16 checkpoint is 131 shards / ~350 GB. We never need more than a
couple of expert matrices, so this reads the safetensors header, computes the byte
range of the region we want, and issues an HTTP Range request for just that.

Cache lives under /data (the big disk), not the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

CACHE = Path(os.environ.get("FLASH_NEXT_SLICE_CACHE", "/data/models/flash-next-bf16-slices"))
REPO = "Qwen/Qwen3.8-Flash-Next"

_DTYPES = {"BF16": (2, "bfloat16"), "F16": (2, "float16"), "F32": (4, "float32")}


def _url(repo: str, filename: str) -> str:
    return hf_hub_url(repo, filename)


def _get_range(url: str, start: int, end: int) -> bytes:
    """Inclusive-exclusive byte range."""
    tok = os.environ.get("HF_TOKEN")
    headers = {"Range": f"bytes={start}-{end - 1}"}
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    r = requests.get(url, headers=headers, timeout=300, allow_redirects=True)
    r.raise_for_status()
    if len(r.content) != end - start:
        raise RuntimeError(f"short range read: got {len(r.content)}, want {end - start}")
    return r.content


def read_header(repo: str, filename: str) -> tuple[dict, int]:
    """Returns (header dict, data section offset)."""
    url = _url(repo, filename)
    n = struct.unpack("<Q", _get_range(url, 0, 8))[0]
    hdr = json.loads(_get_range(url, 8, 8 + n))
    return hdr, 8 + n


def fetch(repo: str, tensor: str, first: int | None = None, count: int | None = None,
          index: dict | None = None) -> np.ndarray:
    """Fetch `tensor`, optionally only [first : first+count] along axis 0.

    Slicing along axis 0 is a contiguous byte range in safetensors' row-major layout,
    which is exactly what an expert index is for `experts.gate_up_proj` [E, ...].
    """
    tag = f"{repo.replace('/', '__')}__{tensor}"
    if first is not None:
        tag += f"__{first}_{count}"
    cached = CACHE / f"{tag}.npy"
    if cached.is_file():
        return np.load(cached)

    CACHE.mkdir(parents=True, exist_ok=True)
    if index is None:
        p = hf_hub_download(repo, "model.safetensors.index.json")
        index = json.load(open(p))["weight_map"]
    filename = index[tensor]
    hdr, data_off = read_header(repo, filename)
    meta = hdr[tensor]
    dtype_name = meta["dtype"]
    if dtype_name not in _DTYPES:
        raise NotImplementedError(f"dtype {dtype_name}")
    itemsize, np_name = _DTYPES[dtype_name]
    shape = list(meta["shape"])
    begin, _end = meta["data_offsets"]

    row_elems = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    row_bytes = row_elems * itemsize
    if first is None:
        first, count = 0, shape[0]
    out_shape = [count] + shape[1:]

    start = data_off + begin + first * row_bytes
    raw = _get_range(_url(repo, filename), start, start + count * row_bytes)

    if np_name == "bfloat16":
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        arr = u16.view(np.float32).reshape(out_shape)
    else:
        arr = np.frombuffer(raw, dtype=f"<{np.dtype(np_name).str[1:]}").reshape(out_shape)
        arr = arr.astype(np.float32)

    np.save(cached, arr)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tensor")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--count", type=int, default=None)
    a = ap.parse_args()
    arr = fetch(a.repo, a.tensor, a.first, a.count)
    print(f"{a.tensor}  shape={arr.shape} dtype={arr.dtype} "
          f"absmax={np.abs(arr).max():.5g} std={arr.std():.5g}")
    print(f"cached under {CACHE}")


if __name__ == "__main__":
    main()
