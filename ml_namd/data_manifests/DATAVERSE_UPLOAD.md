# Dataverse upload manifest

What the Harvard Dataverse record must contain, and how it maps to the examples. The code lives in this
GitHub repository; the data + initial conditions live in the Dataverse record:

> Khanna, Ajay, 2026, *Replication Data for: Machine Learning for Non Adiabatic Excited State Molecular
> Dynamics*, https://doi.org/10.7910/DVN/HCL8EK, Harvard Dataverse.

Only the **base active-learning (BALF)** pools are shipped (GALF and random-split variants are out of
scope). The arrays follow the schema in `DATA_MANIFEST.md`.

---

## Proposed record layout

```
enol/balf/     19 .npy  : acn_R, acn_Z, acn_sE0..3, acn_F0..3,
                          acn_dENACR{12,13,23}, acn_GAP{12,13,23}, acn_GDV{12,13,23}
enol/ics/      enol1ps_R0.npy, enol1ps_V0.npy, enol1ps_Z.npy, PROVENANCE.md
keto/balf/     44 .npy  : acn_R, acn_Z, acn_sE0..5, acn_F0..5,
                          acn_dENACR{12,13,14,15,23,24,25,34,35,45} and the matching acn_GAP{..}/acn_GDV{..}
keto/ics/      keto10ps_R0.npy, keto10ps_V0.npy, keto10ps_Z.npy, PROVENANCE.md
docs/          a copy of the how-to markdowns, so the dataset is self-describing (see below)
SHA256SUMS     checksums for every .npy
```

**Note on the keto pool.** It ships the full S0-S5 / 10-pair superset, but the worked example uses only
the S0-S3 subset (`n_states = 3`, coupling pairs `12, 13, 23`); the extra states/pairs are ignored by the
example config and are provided for users who want to go further. The enol pool is exactly S0-S3.

## How a downloader places the data (matches the examples)

For each tautomer, the example expects the pool under `data/<taut>/balf/` and the ICs under `ics/`:

```
<repo>/ml_namd/examples/enol/data/enol/balf/   <- enol/balf/   from the record
<repo>/ml_namd/examples/enol/ics/              <- enol/ics/*.npy
<repo>/ml_namd/examples/keto/data/keto/balf/   <- keto/balf/
<repo>/ml_namd/examples/keto/ics/              <- keto/ics/*.npy
```

(Each example's `00_download_data.md` says the same.) The IC file prefixes are `enol1ps_` and `keto10ps_`.

## How-to markdowns to include in the record `docs/`

So the dataset is usable without first cloning the code, include copies of:
- the two example READMEs and their `00_download_data.md`,
- `stratified_split_guide.md`, `model_a_guide.md`, `model_b_usage_guide.md`,
- `fssh_mlip_guide.md`, `fssh_pyseqm_guide.md` (the "how to run ML-TSH" + harvesting steps),
- `DATA_MANIFEST.md` (the array schema).

## Integrity
Generate `SHA256SUMS` over every uploaded `.npy` and include it in the record. The initial-condition
`PROVENANCE.md` files already carry their hash-verification; keep them alongside the IC arrays.
