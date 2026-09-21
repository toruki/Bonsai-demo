"""Compare two dump_hidden outputs: per-layer residual-stream error and logit metrics.

    python compare_hidden.py REF_DIR TEST_DIR [--json out.json]

Per layer l (l_out-l): cosine(mean over tokens), relative MSE (||a-b||^2/||a||^2,
summed over tokens). Logits: mean token-level KL(ref || test), top-1 / top-5
agreement, mean logit cosine, and a log-prob-of-ref-argmax drop.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def load(d: Path, name: str) -> np.ndarray:
    meta = json.load(open(d / "meta.json"))
    ne0, nt = meta["tensors"][name]
    return np.fromfile(d / f"{name}.f32", dtype=np.float32).reshape(nt, ne0)


def layer_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    num = (a64 * b64).sum(axis=1)
    den = np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1) + 1e-30
    cos = num / den
    err = ((a64 - b64) ** 2).sum()
    return {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
            "rel_mse": float(err / (a64 ** 2).sum()), "ref_rms": float(np.sqrt((a64 ** 2).mean()))}


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def logit_metrics(a: np.ndarray, b: np.ndarray, skip_first: int = 1) -> dict:
    a, b = a[skip_first:].astype(np.float64), b[skip_first:].astype(np.float64)
    la, lb = log_softmax(a), log_softmax(b)
    pa = np.exp(la)
    kl = (pa * (la - lb)).sum(axis=1)
    top1a, top1b = a.argmax(axis=1), b.argmax(axis=1)
    top5a = np.argsort(-a, axis=1)[:, :5]
    top5b = np.argsort(-b, axis=1)[:, :5]
    top5 = np.array([len(set(x) & set(y)) / 5 for x, y in zip(top5a, top5b)])
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)
    # probability the test model assigns to the ref model's argmax
    p_ref_argmax = np.exp(lb[np.arange(len(top1a)), top1a])
    return {"kl_mean": float(kl.mean()), "kl_median": float(np.median(kl)), "kl_p99": float(np.percentile(kl, 99)),
            "top1_agree": float((top1a == top1b).mean()), "top5_overlap": float(top5.mean()),
            "logit_cos_mean": float(cos.mean()),
            "test_prob_of_ref_top1_mean": float(p_ref_argmax.mean()), "n_positions": int(len(kl))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("test")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ref, test = Path(a.ref), Path(a.test)
    mr, mt = json.load(open(ref / "meta.json")), json.load(open(test / "meta.json"))
    assert mr["tokens"] == mt["tokens"], "token streams differ; models must share the tokenizer/prompt"
    layers = sorted(int(m.group(1)) for k in mr["tensors"] if (m := re.match(r"l_out-(\d+)$", k)))
    out = {"layers": {}, "n_tokens": mr["n_tokens"]}
    print(f"{'layer':>6s} {'cos_mean':>9s} {'cos_min':>8s} {'rel_mse':>9s} {'ref_rms':>8s}")
    for l in layers:
        m = layer_metrics(load(ref, f"l_out-{l}"), load(test, f"l_out-{l}"))
        out["layers"][l] = m
        print(f"{l:6d} {m['cos_mean']:9.5f} {m['cos_min']:8.4f} {m['rel_mse']:9.5f} {m['ref_rms']:8.2f}")
    if "result_norm" in mr["tensors"] and "result_norm" in mt["tensors"]:
        m = layer_metrics(load(ref, "result_norm"), load(test, "result_norm"))
        out["result_norm"] = m
        print(f"{'norm':>6s} {m['cos_mean']:9.5f} {m['cos_min']:8.4f} {m['rel_mse']:9.5f}")
    lm = logit_metrics(load(ref, "logits"), load(test, "logits"))
    out["logits"] = lm
    print("\nlogits:", json.dumps(lm, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
