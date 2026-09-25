"""Offline replay of a global (layer, expert) LRU expert cache over recorded routing.

Input: a dump_hidden directory with ffn_moe_topk-N.i32 ([tokens, k] per layer, tokens in corpus
order). For each token the layers are visited in order and each selected expert is looked up in one
global LRU of S slots (the FreeToken design: one pool shared by all cached layers). Reports the hit
rate and the miss traffic in MiB per token for several slot budgets and bytes-per-expert.

    python lru_replay.py TRACE_DIR --layers 4 47 --slots 22528 19456 14848 --bytes-per-expert 1.025
"""
import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--layers", type=int, nargs=2, default=[4, 47], help="first and last cached layer")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--slots", type=int, nargs="+", default=[22528, 19456, 14848, 11264, 7424])
    ap.add_argument("--bytes-per-expert", type=float, default=1.025, help="MiB per (layer, expert) slot")
    ap.add_argument("--warmup", type=int, default=2048, help="tokens excluded from the statistics")
    a = ap.parse_args()
    d = Path(a.trace)
    L0, L1 = a.layers
    routes = np.stack([np.fromfile(d / f"ffn_moe_topk-{l}.i32", np.int32).reshape(-1, a.k) for l in range(L0, L1 + 1)], axis=1)
    T, L, _ = routes.shape
    total = L * 512
    print(f"{T} tokens, {L} cached layers x 512 experts = {total} slots total; k={a.k}")
    # popularity: how concentrated is the routing?
    for l in (L0, (L0 + L1) // 2, L1):
        cnt = np.bincount(routes[:, l - L0].reshape(-1), minlength=512); cnt = np.sort(cnt)[::-1]
        cum = np.cumsum(cnt) / cnt.sum()
        print(f"  layer {l}: experts covering 50% / 90% of the picks: {int(np.searchsorted(cum, 0.5)) + 1} / {int(np.searchsorted(cum, 0.9)) + 1}")
    print(f"{'slots':>8s} {'frac':>6s} {'hit':>7s} {'miss/tok':>9s} {'MiB/tok':>8s}")
    for s in a.slots:
        # one pass; the first `warmup` tokens fill the cache and are excluded from the statistics
        cache = OrderedDict(); hits = misses = 0
        for t in range(T):
            for l in range(L):
                for e in routes[t, l]:
                    key = l * 1024 + int(e)
                    if key in cache:
                        cache.move_to_end(key)
                        if t >= a.warmup: hits += 1
                    else:
                        if t >= a.warmup: misses += 1
                        cache[key] = None
                        if len(cache) > s: cache.popitem(last=False)
        n = max(T - a.warmup, 1)
        print(f"{s:8d} {s / total:6.2f} {hits / (hits + misses):7.3f} {misses / n:9.1f} {misses / n * a.bytes_per_expert:8.1f}")


if __name__ == "__main__":
    main()
