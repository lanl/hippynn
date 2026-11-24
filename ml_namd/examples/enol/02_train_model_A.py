#!/usr/bin/env python3
"""Step 2: train Model A (E+F) for one seed, then run genA (the A-predicted GDV/GAP pool).

Stage 1a generates + saves the gap-stratified split masks and trains Model A. Stage 1b runs the frozen
Model A over the pool to write the A-predicted GDV/GAP that Model B will train on (so training matches
inference), stamping genA_source.json (the train-A == deploy-A record).

Usage: python 02_train_model_A.py --seed 42 [--device auto]
"""
import argparse
import subprocess
import sys

import _common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    env = C.base_env()
    sys.path.insert(0, str(C.TOOLS))
    import config_gen

    cfg_dir = C.RUN_ROOT / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg_dir / f"A_{C.TAUT}_balf_stratified_seed{a.seed}.inp"
    cfg.write_text(config_gen.gen_A(C.TAUT, "BALF", a.seed, "stratified"))
    print(f"[02] wrote {cfg}", flush=True)

    # Stage 1a: train Model A (run_train_rndsplit seeds the init + injects the saved stratified split)
    subprocess.run([sys.executable, str(C.TOOLS / "run_train_rndsplit.py"),
                    "--config_file", str(cfg),
                    "--trainer-dir", str(C.TRAINER),
                    "--tools-dir", str(C.TOOLS)], env=env, check=True)

    # Stage 1b: genA -- frozen Model A -> A-predicted GDV/GAP pool for Model B
    gen_env = dict(env)
    gen_env["MODELA_DIR"] = str(C.modelA_dir(a.seed))
    gen_env["DATA_DIR"] = str(C.pool_dir())
    gen_env["MODELB_OUT"] = str(C.genA_pool(a.seed))
    gen_env["GENA_TRAIN_DIR"] = str(C.TRAINER)
    subprocess.run([sys.executable, str(C.DATAGEN / "gen_modelA_inputs.py"),
                    "--tautomer", C.TAUT, "--device", a.device], env=gen_env, check=True)

    print(f"[02] DONE seed={a.seed}: Model A at {C.modelA_dir(a.seed)}; "
          f"genA pool at {C.genA_pool(a.seed)}", flush=True)


if __name__ == "__main__":
    main()
