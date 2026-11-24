#!/usr/bin/env python3
"""Config generator + verifier for the 5-random-split Model-A/Model-B retraining campaign.

SINGLE SOURCE OF TRUTH = SPEC below (the user's exact hyperparameters). Every config inherits
the identical shared block by construction, so the hyperparameters are guaranteed EXACTLY as
provided and IDENTICAL between tautomers. The ONLY permitted per-config differences are:
  - data_dir / output_dir / seed            (always)
  - state_nacr_weights  (keto B only = 12:4.0; enol B omits it -> uniform)   [Model B]
  - direct_nacr_v2_mode (gdv_only/both/geom_only)                            [Model B head]
Anything else differing between keto and enol is a BUG -> verify() asserts it cannot happen.

Run: python config_gen.py [--emit <outdir>]   (default: verify-only, prints the proof table)
"""
import argparse, os, sys

# ============================ USER SPEC (single source of truth) ============================
# Same for Model A and Model B, same for keto and enol:
SPEC = {
    "n_states":             "3",                    # S0-S3
    "n_features":           "145",
    "n_sensitivities":      "20",
    "n_interaction_layers": "2",
    "n_atom_layers":        "3",
    "l_max":                "3",
    "n_max":                "4",
    "dist_soft_min":        "0.7",
    "dist_soft_max":        "5.67638290906325",     # = 5.68
    "dist_hard_max":        "11.833504244685173",   # = 11.83
    "learning_rate":        "4.31e-4",
    "batch_size":           "64",
    "max_batch_size":       "64",                   # = no batch raising (Batch Size = 64)
    "max_epochs":           "300",
    "l2_weight":            "2e-5",
    "termination_patience": "50",
    "scheduler_patience":   "16",
    "scheduler_factor":     "0.2",
    "test_size":            "0.1",                  # 80/10/10
    "valid_size":           "0.1",
}
# shared settings NOT in the user's list but required + identical across tautomers (recipe defaults):
SHARED_COMMON = {
    "network_type": "hiphopnn", "possible_species": "0 1 6 8", "n_atoms": "15",
    "dataset_prefix": "acn_", "fraction_train_eval": "0.1", "plot_frequency": "50",
    "train_tdipoles": "false",
}
# Data + run roots are set for your system (the defaults are placeholders). The examples point these
# at the Dataverse-downloaded pools under examples/<taut>/data/<cell> and the example run directories.
_DATA_ROOT = os.environ.get("ML_NAMD_DATA_ROOT", "<DATA_ROOT>")
POOL = {
    (t, c): f"{_DATA_ROOT}/{t}/{c.lower()}"
    for t in ("keto", "enol") for c in ("BALF", "GALF")
}
HEAD_V2MODE = {"gdv_only": "gdv_pair_mixing", "both": "gdv_pair_mixing_residual", "geom_only": "geom_residual_only"}
RUNROOT = os.environ.get("ML_NAMD_RUN_ROOT", "<RUN_ROOT>")


def _block(d):
    return "\n".join(f"{k:<28}= {v}" for k, v in d.items())


def gen_A(taut, cell, seed, split):
    d = dict(SPEC); d.update(SHARED_COMMON)
    d.update({  # Model-A specific (identical both tautomers)
        "train_energies": "true", "train_forces": "true", "train_scaled_nacrs": "false",
        "include_ground_state": "true", "force_sign": "-1",
        "energy_weight": "1.0", "force_weight": "1.0", "stopping_key": "Loss",
        "eval_batch_size": "256",
        "seed": str(seed),
        "data_dir": POOL[(taut, cell)],
        "output_dir": f"{RUNROOT}/model_A/{taut}/{cell.lower()}/{split}/rnd_seed_{seed}/model",
    })
    return "# Model A (E+F, S0-S3) :: %s %s %s seed=%s :: 5-rnd-split campaign (unified spec)\n%s\n" % (
        taut, cell, split, seed, _block(d))


def gen_B(taut, cell, head, seed, split):
    d = dict(SPEC); d.update(SHARED_COMMON)
    d.update({  # Model-B specific (identical both tautomers except the weighting)
        "train_energies": "false", "train_forces": "false", "train_scaled_nacrs": "true",
        "scaled_nacr_head": "direct_vector_v3", "targeted_scaled_nacr_pairs": "12, 13, 23",
        "direct_nacr_v2_mode": HEAD_V2MODE[head], "direct_nacr_gap_gate_mode": "learned",
        "gdv_source": "file", "gdv_file_template": "{prefix}GDV{pair}.npy",
        "gap_file_template": "{prefix}GAP{pair}.npy", "phasesign_corrected_scaled_nacr": "false",
        "nacr_weight": "1.0", "eval_batch_size": "32",
        "seed": str(seed),
        # genA_pool is HEAD-INDEPENDENT (frozen A depends only on cell,split,seed) -> ONE pool
        # per (cell,split,seed), shared by all 3 B heads so they train on the identical A-pred inputs.
        "data_dir": f"{RUNROOT}/model_B/{taut}/{cell.lower()}/{split}/rnd_seed_{seed}/genA_pool",
        "output_dir": f"{RUNROOT}/model_B/{taut}/{cell.lower()}/{head}/{split}/rnd_seed_{seed}/model",
    })
    if taut == "keto":                       # keto ONLY: w12=4, else 1 ; enol = uniform (omit)
        d["state_nacr_weights"] = "12:4.0"
    return "# Model B (dENACR v3, head=%s, S0-S3) :: %s %s %s seed=%s :: keto w12=4 / enol uniform\n%s\n" % (
        head, taut, cell, split, seed, _block(d))


def _parse(cfgstr):
    out = {}
    for line in cfgstr.splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1); out[k.strip()] = v.strip()
    return out


def verify():
    ok = True
    # 1) every SPEC key appears with the EXACT value in A and B, both tautomers
    for mk, gen in (("A", gen_A), ("B", gen_B)):
        for taut in ("keto", "enol"):
            c = _parse(gen(taut, "BALF", 42, "random") if mk == "A" else gen(taut, "BALF", "both", 42, "random"))
            for k, v in SPEC.items():
                if c.get(k) != v:
                    print(f"  SPEC MISMATCH [{mk} {taut}] {k}: config={c.get(k)!r} spec={v!r}"); ok = False
    # 2) keto vs enol IDENTICAL except the allowed keys
    allowed_A = {"data_dir", "output_dir", "seed"}
    allowed_B = {"data_dir", "output_dir", "seed", "state_nacr_weights"}
    for mk, gen, allowed, extra in (("A", gen_A, allowed_A, ()), ("B", gen_B, allowed_B, ("both",))):
        ck = _parse(gen("keto", "BALF", 42, "random", *extra) if mk == "A" else gen("keto", "BALF", "both", 42, "random"))
        ce = _parse(gen("enol", "BALF", 42, "random", *extra) if mk == "A" else gen("enol", "BALF", "both", 42, "random"))
        keys = set(ck) | set(ce)
        diffs = {k for k in keys if ck.get(k) != ce.get(k)}
        bad = diffs - allowed
        print(f"  [{mk}] keto-vs-enol differ in: {sorted(diffs)}  (allowed: {sorted(allowed)})")
        if bad:
            print(f"    UNEXPECTED cross-tautomer difference: {sorted(bad)}"); ok = False
    # 3) the 3 B heads differ ONLY in direct_nacr_v2_mode
    base = _parse(gen_B("keto", "BALF", "both", 42, "random"))
    for h in ("gdv_only", "geom_only"):
        ch = _parse(gen_B("keto", "BALF", h, 42, "random"))
        diffs = {k for k in (set(base) | set(ch)) if base.get(k) != ch.get(k)}
        print(f"  [B head both-vs-{h}] differ in: {sorted(diffs)}")
        if diffs != {"direct_nacr_v2_mode", "output_dir"}:   # data_dir now SHARED across heads
            print(f"    UNEXPECTED head difference: {sorted(diffs - {'direct_nacr_v2_mode','output_dir'})}"); ok = False
    print("\nVERIFY:", "PASS — hyperparameters exactly match the spec and are identical between tautomers" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--emit", default=""); a = ap.parse_args()
    print("=" * 70); print("SPEC (single source of truth):")
    for k, v in SPEC.items():
        print(f"  {k:<22}= {v}")
    print("=" * 70)
    print("\n----- EXAMPLE: keto A (BALF seed42 random) -----\n" + gen_A("keto", "BALF", 42, "random"))
    print("----- EXAMPLE: keto B both (BALF seed42 random) -----\n" + gen_B("keto", "BALF", "both", 42, "random"))
    print("=" * 70); print("VERIFICATION:")
    ok = verify()
    if a.emit and ok:
        SPLITS = ("random", "stratified"); SEEDS = (42, 101726, 102593, 112793, 112891)
        HEADS = ("gdv_only", "both", "geom_only"); n = 0
        for taut in ("keto", "enol"):
            for cell in ("BALF", "GALF"):
                for sp in SPLITS:
                    for s in SEEDS:
                        p = f"{a.emit}/A_{taut}_{cell.lower()}_{sp}_seed{s}.inp"
                        open(p, "w").write(gen_A(taut, cell, s, sp)); n += 1
                        for h in HEADS:
                            p = f"{a.emit}/B_{taut}_{cell.lower()}_{h}_{sp}_seed{s}.inp"
                            open(p, "w").write(gen_B(taut, cell, h, s, sp)); n += 1
        print(f"\nEMITTED {n} configs to {a.emit}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
