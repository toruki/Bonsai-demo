"""Bit-exact Python mirror of the Prism ternary packing formats and rotated basis.

Everything here is transcribed from the fork's C sources, not reinvented:

  PQ2_0   ggml/src/ggml-quants.c :: quantize_row_pq2_0_ref / dequantize_row_pq2_0
  PTQ1_0  ggml/src/ggml-quants.c :: quantize_row_ptq1_0_ref / dequantize_row_ptq1_0
  block layouts             ggml/src/ggml-common.h :: block_pq2_0 / block_ptq1_0
  rotation matrix R         src/llama-model.cpp (llama_model::load_tensors, the
                            "prism.hadamard" block that fills a block_size^2 F32 tensor)
  activation-side transform src/llama-graph.cpp :: build_lora_mm / build_lora_mm_id
                            + src/llama-impl.h :: llama_mul_mat_hadamard

`selftest` re-packs tensors taken straight out of the shipped Bonsai 2 27B GGUF and
asserts the bytes come back identical, so the transcription is checked, not assumed.
"""

from __future__ import annotations

import numpy as np

QK_PQ2_0 = 128
QK_PTQ1_0 = 128
PQ2_0_BLOCK_BYTES = 2 + QK_PQ2_0 // 4          # 34
PTQ1_0_BLOCK_BYTES = 24 + 2 + 2                # 28
PTQ1_0_STAGES = (32, 16, 8)                    # ggml-quants.c :: ptq1_0_stages


# --------------------------------------------------------------------------- PQ2_0

def pq2_0_quantize(x: np.ndarray) -> np.ndarray:
    """float32 [..., k] -> uint8 [..., k/128, 34]. Mirrors quantize_row_pq2_0_ref."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, QK_PQ2_0)
    nb = x.shape[0]

    d = np.abs(x).max(axis=1)                       # const float d = amax;
    d16 = d.astype(np.float16)
    dq = d16.astype(np.float32)
    idq = np.where(dq > 0.0, 1.0 / np.where(dq == 0.0, 1.0, dq), 0.0).astype(np.float32)

    # NOTE: the C code computes id from the *fp32* amax, then stores the fp16 of it.
    id_ = np.where(d > 0.0, 1.0 / np.where(d == 0.0, 1.0, d), 0.0).astype(np.float32)
    q = np.rint(x * id_[:, None]).astype(np.int32) + 1
    np.clip(q, 0, 3, out=q)

    out = np.zeros((nb, PQ2_0_BLOCK_BYTES), dtype=np.uint8)
    out[:, :2] = d16.view(np.uint8).reshape(nb, 2)
    qs = np.zeros((nb, QK_PQ2_0 // 4), dtype=np.uint8)
    for j in range(QK_PQ2_0):
        qs[:, j // 4] |= (q[:, j].astype(np.uint8) << ((j % 4) * 2))
    out[:, 2:] = qs
    _ = idq  # kept only to document that dequant uses the fp16-rounded scale
    return out


def pq2_0_dequantize(blocks: np.ndarray) -> np.ndarray:
    """uint8 [nb, 34] -> float32 [nb*128]. Mirrors dequantize_row_pq2_0."""
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8).reshape(-1, PQ2_0_BLOCK_BYTES)
    nb = blocks.shape[0]
    d = blocks[:, :2].copy().view(np.float16).reshape(nb).astype(np.float32)
    qs = blocks[:, 2:]
    codes = np.empty((nb, QK_PQ2_0), dtype=np.int8)
    for j in range(QK_PQ2_0):
        codes[:, j] = (qs[:, j // 4] >> ((j % 4) * 2)) & 3
    return ((codes.astype(np.float32) - 1.0) * d[:, None]).reshape(-1)


def pq2_0_codes(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """uint8 [nb, 34] -> (trits int8 [nb, 128] in {-1,0,1,2}, scale float32 [nb])."""
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8).reshape(-1, PQ2_0_BLOCK_BYTES)
    nb = blocks.shape[0]
    d = blocks[:, :2].copy().view(np.float16).reshape(nb).astype(np.float32)
    qs = blocks[:, 2:]
    codes = np.empty((nb, QK_PQ2_0), dtype=np.int8)
    for j in range(QK_PQ2_0):
        codes[:, j] = ((qs[:, j // 4] >> ((j % 4) * 2)) & 3).astype(np.int8) - 1
    return codes, d


# -------------------------------------------------------------------------- PTQ1_0

def _ptq1_0_positions() -> tuple[np.ndarray, np.ndarray]:
    """Element index -> (byte slot) map for qs and qh, straight from the ref loops.

    Returns (qs_map, qh_map) where qs_map[j, n] is the source element index packed
    into qs byte j at trit position n (n=0 is the most significant trit), and
    qh_map[h, m] likewise for qh.
    """
    qs_map = np.zeros((24, 5), dtype=np.int64)
    consumed = 0
    j = 0
    for c in PTQ1_0_STAGES:
        while j + c <= 24:
            for m in range(c):
                for n in range(5):
                    qs_map[j + m, n] = consumed + m + n * c
            consumed += 5 * c
            j += c
    qh_map = np.zeros((2, 4), dtype=np.int64)
    for h in range(2):
        for m in range(4):
            qh_map[h, m] = consumed + h + m * 2
    return qs_map, qh_map


_QS_MAP, _QH_MAP = _ptq1_0_positions()


def ptq1_0_quantize(x: np.ndarray) -> np.ndarray:
    """float32 [..., k] -> uint8 [..., k/128, 28]. Mirrors quantize_row_ptq1_0_ref."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, QK_PTQ1_0)
    nb = x.shape[0]

    d = np.abs(x).max(axis=1)
    id_ = np.where(d > 0.0, 1.0 / np.where(d == 0.0, 1.0, d), 0.0).astype(np.float32)
    xi = np.rint(x * id_[:, None]).astype(np.int32) + 1     # 0,1,2

    out = np.zeros((nb, PTQ1_0_BLOCK_BYTES), dtype=np.uint8)

    q = np.zeros((nb, 24), dtype=np.uint32)
    for n in range(5):
        q = q * 3 + xi[:, _QS_MAP[:, n]]
    out[:, 0:24] = ((q * 256 + 242) // 243).astype(np.uint8)

    qh = np.zeros((nb, 2), dtype=np.uint32)
    for m in range(4):
        qh = qh * 3 + xi[:, _QH_MAP[:, m]]
    qh = qh * 3                                              # shift to the MS trit
    out[:, 24:26] = ((qh * 256 + 242) // 243).astype(np.uint8)

    out[:, 26:28] = d.astype(np.float16).view(np.uint8).reshape(nb, 2)
    return out


def ptq1_0_dequantize(blocks: np.ndarray) -> np.ndarray:
    """uint8 [nb, 28] -> float32 [nb*128]. Mirrors dequantize_row_ptq1_0."""
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8).reshape(-1, PTQ1_0_BLOCK_BYTES)
    nb = blocks.shape[0]
    d = blocks[:, 26:28].copy().view(np.float16).reshape(nb).astype(np.float32)
    pow3 = (1, 3, 9, 27, 81, 243)

    out = np.zeros((nb, QK_PTQ1_0), dtype=np.float32)
    qs = blocks[:, 0:24].astype(np.uint16)
    for n in range(5):
        v = ((qs * pow3[n]) & 0xFF)
        xi = ((v * 3) >> 8).astype(np.int16) - 1
        out[:, _QS_MAP[:, n]] = xi.astype(np.float32)
    qh = blocks[:, 24:26].astype(np.uint16)
    for n in range(4):
        v = ((qh * pow3[n]) & 0xFF)
        xi = ((v * 3) >> 8).astype(np.int16) - 1
        out[:, _QH_MAP[:, n]] = xi.astype(np.float32)
    return (out * d[:, None]).reshape(-1)


# ------------------------------------------------------------------- rotated basis

def hadamard_matrix(n: int) -> np.ndarray:
    """Normalized Sylvester-Walsh H_n, exactly as llama-model.cpp builds it.

    data[row*n + col] = (popcount(row & col) & 1) ? -1/sqrt(n) : +1/sqrt(n)
    """
    if n & (n - 1):
        raise ValueError(f"block size {n} is not a power of two")
    r = np.arange(n, dtype=np.uint32)[:, None]
    c = np.arange(n, dtype=np.uint32)[None, :]
    parity = np.bitwise_count(r & c) & 1 if hasattr(np, "bitwise_count") else None
    if parity is None:                                        # numpy < 2.0
        v = (r & c).astype(np.uint32)
        parity = np.zeros_like(v)
        while v.any():
            parity ^= v & 1
            v >>= 1
        parity &= 1
    h = np.where(parity.astype(bool), -1.0, 1.0).astype(np.float32)
    return h / np.float32(np.sqrt(n))


def rotate_activation(x: np.ndarray, block: int, signs: np.ndarray | None) -> np.ndarray:
    """R x with R = (1/sqrt(n)) H_n S, applied blockwise along the last axis.

    This is the runtime path: ggml_mul(cur, signs) then a mul_mat against the
    block_size x block_size rotation tensor over a [block_size, rest] reshape.
    """
    x = np.asarray(x, dtype=np.float32)
    if signs is not None:
        x = x * np.asarray(signs, dtype=np.float32)
    h = hadamard_matrix(block)
    shp = x.shape
    xb = x.reshape(-1, block)
    # ggml_mul_mat(rot, res) contracts rot's ne[0] with res's ne[0]: y[i] = sum_j H[i][j] x[j]
    return (xb @ h.T).reshape(shp)


def fold_weight(w: np.ndarray, block: int, signs: np.ndarray | None) -> np.ndarray:
    """W -> W_folded such that W_folded @ (R x) == W @ x.

    R = H S (S = diag(signs), H the normalized symmetric Walsh-Hadamard block), so
    W_folded = W R^{-1} = W S^{-1} H^{-1} = W S H: scale the columns by the signs
    first, then apply H blockwise along the input axis. The order matters -- H is
    blockwise and S is not diagonal in the same basis after rotation.
    """
    w = np.asarray(w, dtype=np.float32)
    n_in = w.shape[-1]
    if n_in % block:
        raise ValueError(f"input dim {n_in} not divisible by block {block}")
    if signs is not None:
        w = w * np.asarray(signs, dtype=np.float32)
    h = hadamard_matrix(block)
    return (w.reshape(-1, block) @ h).reshape(w.shape)


def unfold_weight(wf: np.ndarray, block: int, signs: np.ndarray | None) -> np.ndarray:
    """Inverse of fold_weight: W = (W_folded H) S."""
    wf = np.asarray(wf, dtype=np.float32)
    h = hadamard_matrix(block)
    w = (wf.reshape(-1, block) @ h).reshape(wf.shape)
    if signs is not None:
        w = w * np.asarray(signs, dtype=np.float32)
    return w


# ------------------------------------------------------------------------- selftest

def selftest(gguf_path: str, tensors: tuple[str, ...] = ("blk.0.ffn_gate.weight",),
             rows: int = 64) -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "llama.cpp" / "gguf-py"))
    from gguf import GGUFReader                                   # noqa: E402

    r = GGUFReader(gguf_path)
    by_name = {t.name: t for t in r.tensors}

    # 1. Hadamard identities
    for n in (128, 1024):
        h = hadamard_matrix(n)
        assert np.allclose(h @ h.T, np.eye(n), atol=1e-5), f"H_{n} not orthonormal"
        assert np.array_equal(h, h.T), f"H_{n} not symmetric"
    rng = np.random.default_rng(0)
    w = rng.standard_normal((37, 2048)).astype(np.float32)
    x = rng.standard_normal((5, 2048)).astype(np.float32)
    s = rng.choice(np.array([-1.0, 1.0], np.float32), 2048)
    lhs = rotate_activation(x, 1024, s) @ fold_weight(w, 1024, s).T
    assert np.allclose(lhs, x @ w.T, rtol=1e-4, atol=1e-4), "fold/rotate identity broken"
    assert np.allclose(unfold_weight(fold_weight(w, 1024, s), 1024, s), w, atol=1e-5)
    print("hadamard identities                       OK")

    # 2. byte-exact repack of real shipped weights
    for name in tensors:
        t = by_name[name]
        ne0 = int(t.shape[0])
        nblk_row = ne0 // QK_PQ2_0
        raw = t.data.view(np.uint8).reshape(-1, PQ2_0_BLOCK_BYTES)[: rows * nblk_row]
        f32 = pq2_0_dequantize(raw)

        re_pq = pq2_0_quantize(f32)
        assert np.array_equal(re_pq, raw), f"PQ2_0 repack mismatch on {name}"

        pt = ptq1_0_quantize(f32)
        back = ptq1_0_dequantize(pt)
        assert np.array_equal(back, f32), f"PTQ1_0 round-trip mismatch on {name}"
        print(f"{name:32s} PQ2_0 repack byte-exact / PTQ1_0 lossless "
              f"({raw.shape[0]} blocks)")


if __name__ == "__main__":
    import sys
    selftest(sys.argv[1] if len(sys.argv) > 1
             else "models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf",
             tensors=("blk.0.ffn_gate.weight", "blk.0.attn_qkv.weight", "blk.0.ssm_out.weight"))
