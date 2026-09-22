"""Regenerate the student stream at a layer boundary from saved student states.

    python regen_stream.py --upto 60 --out /data/eval/prog_dual      -> S_60_{train,valid}.f16
Also checks the regenerated stream at the last saved boundary against the saved one.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from progressive import ACT_DIRS, VALID_DIR, Rotary, embed, load_f16, run_layers, save_f16, tokens_of  # noqa: E402
from student_utils import load_student_layer  # noqa: E402
from ternary_layer import load_config, recon_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, required=True, help="boundary index: stream entering this layer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", type=int, default=64, help="compare with saved S_{check}_valid if reachable")
    a = ap.parse_args()
    out = Path(a.out); cfg = load_config(); rotary = Rotary(cfg)
    tok_tr = torch.cat([tokens_of(d) for d in ACT_DIRS]); tok_va = tokens_of(VALID_DIR)
    S_tr, S_va = embed(tok_tr), embed(tok_va)
    t0 = time.time()
    last = max(a.upto, a.check if (out / f"S_{a.check}_valid.f16").exists() else a.upto)
    for k in range(0, last):
        mod, _ = load_student_layer(k, out)
        S_va = run_layers([mod], S_va, rotary)
        if k < a.upto:
            S_tr = run_layers([mod], S_tr, rotary)
        if k + 1 == a.upto:
            save_f16(S_tr, out / f"S_{a.upto}_train.f16"); save_f16(S_va, out / f"S_{a.upto}_valid.f16")
            print(f"saved S_{a.upto} ({time.time()-t0:.0f}s)", flush=True)
        del mod; torch.cuda.empty_cache()
        if (k + 1) % 8 == 0:
            print(f"  layer {k} done ({time.time()-t0:.0f}s)", flush=True)
    if last == a.check and (out / f"S_{a.check}_valid.f16").exists():
        ref = load_f16(out / f"S_{a.check}_valid.f16")
        m = recon_metrics(ref.float(), S_va.float())
        print(f"CHECK regenerated S_{a.check}_valid vs saved: cos {m['cos_mean']:.6f} relMSE {m['rel_mse']:.2e}")
    print("REGEN DONE")


if __name__ == "__main__":
    main()
