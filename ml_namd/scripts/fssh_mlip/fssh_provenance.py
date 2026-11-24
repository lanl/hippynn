#!/usr/bin/env python3
"""L-155 machine enforcement: train-A == deploy-A for composed FSSH@MLIP.

The composed FSSH@MLIP deploys Model A (--modelA-dir) + Model B. Model B was
teacher-forced (L-121) on the GDV *predicted by some Model A* (the genA source).
Correctness REQUIRES that genA-source Model A == the deployed Model A (L-155).
Nothing in the pipeline recorded that link, so sessions kept violating it.

This tool writes the link down and enforces it by hash:

  stamp  : record which Model A's genA produced a Model B training pool.
           writes <pool-dir>/genA_source.json  = {modelA_dir, best_model_sha256, stamped}.
           Called at genA time (auto, patched into gen_modelA_inputs*.py) or retroactively.

  assert : before an FSSH launch, verify the DEPLOY Model A == the pool's stamped
           genA source. exit 1 (aborts a `set -e` slurm BEFORE the driver runs) on
           mismatch; loud WARN (exit 0) if the pool has no stamp (so legacy pools do
           not hard-block, but the gap is visible).

Dependency-free (stdlib only); py3.
"""
import argparse, hashlib, json, os, sys, time


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def best_model(model_dir):
    p = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(p):
        sys.exit("[provenance] ERROR: no best_model.pt in %s" % model_dir)
    return p


def main():
    ap = argparse.ArgumentParser(description="train-A==deploy-A (L-155) stamp/assert")
    ap.add_argument("mode", choices=["stamp", "assert"])
    ap.add_argument("--modelA-dir", required=True,
                    help="genA-source Model A (stamp) / the DEPLOY Model A (assert)")
    ap.add_argument("--pool-dir", required=True,
                    help="Model B training pool (the .inp data_dir); holds/gets genA_source.json")
    a = ap.parse_args()

    stamp_path = os.path.join(a.pool_dir, "genA_source.json")
    ha = sha(best_model(a.modelA_dir))
    real_a = os.path.realpath(a.modelA_dir)

    if a.mode == "stamp":
        rec = {"modelA_dir": real_a, "best_model_sha256": ha,
               "stamped": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        with open(stamp_path, "w") as f:
            json.dump(rec, f, indent=2)
        print("[provenance] STAMPED %s\n  modelA=%s\n  sha256=%s" % (stamp_path, real_a, ha))
        return

    # assert
    if not os.path.exists(stamp_path):
        print("[provenance] WARNING: no genA_source.json in %s -- cannot verify "
              "train-A==deploy-A (L-155). deploy-A=%s sha256=%s. Proceeding UNVERIFIED; "
              "stamp the pool to enable the check." % (a.pool_dir, real_a, ha))
        return
    with open(stamp_path) as f:
        rec = json.load(f)
    if rec.get("best_model_sha256") != ha:
        sys.exit("[provenance] *** L-155 VIOLATION: train-A != deploy-A ***\n"
                 "  deploy Model A : %s  sha256=%s\n"
                 "  genA-source A  : %s  sha256=%s\n"
                 "  Model B was teacher-forced on a DIFFERENT Model A's GDV. Retrain Model B on "
                 "the deploy-A's genA pool, or deploy the matching A. Refusing to launch."
                 % (real_a, ha, rec.get("modelA_dir"), rec.get("best_model_sha256")))
    print("[provenance] OK: train-A == deploy-A (L-155)\n  deploy=%s\n  genA-source=%s\n  sha256=%s"
          % (real_a, rec.get("modelA_dir"), ha))


if __name__ == "__main__":
    main()
