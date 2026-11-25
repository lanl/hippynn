#!/usr/bin/env python3
"""Step 3: train Model B (decoupled dENACR head) for one seed + head, reusing Model A's exact split.

Model B trains on the A-predicted GDV/GAP written by genA (step 2) and reuses Model A's train/valid/test
partition (--reuse-masks-from), so the same geometry's energies/forces and couplings stay together.

Usage: python 03_train_model_B.py --seed 42 [--head both]   # 'both' is the hybrid main-result head
"""
import argparse
import subprocess
import sys

import _common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--head", default="both", choices=C.HEADS,
                    help="'both' = hybrid (main result); 'gdv_only' / 'geom_only' are the ablation heads")
    a = ap.parse_args()

    env = C.base_env()
    sys.path.insert(0, str(C.TOOLS))
    import config_gen

    cfg_dir = C.RUN_ROOT / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg_dir / f"B_{C.TAUT}_balf_{a.head}_stratified_seed{a.seed}.inp"
    cfg.write_text(config_gen.gen_B(C.TAUT, "BALF", a.head, a.seed, "stratified"))
    print(f"[03] wrote {cfg}", flush=True)

    subprocess.run([sys.executable, str(C.TOOLS / "run_train_rndsplit.py"),
                    "--config_file", str(cfg),
                    "--trainer-dir", str(C.TRAINER),
                    "--tools-dir", str(C.TOOLS),
                    "--reuse-masks-from", str(C.modelA_leaf(a.seed))], env=env, check=True)

    print(f"[03] DONE seed={a.seed} head={a.head}: Model B at {C.modelB_dir(a.seed, a.head)}", flush=True)


if __name__ == "__main__":
    main()
