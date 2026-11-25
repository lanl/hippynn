# Running the pipeline (and optional HPC)

The ml_namd pipeline runs as a sequence of plain `python` steps -- **no batch scheduler is required**.
A CUDA GPU is needed for training and dynamics; inspection and small CPU smokes run anywhere. For users
who prefer to submit steps to a Slurm cluster, a generic placeholder template is provided in
`../optional_hpc/` (nothing in it is tied to a specific site).

The end-to-end, copy-pasteable commands for each tautomer live in the worked examples:
`../examples/enol/` (default) and `../examples/keto/`. This document is the operational overview behind
them.

---

## 1. Stages

| Stage | What runs | Script |
|---|---|---|
| Data split | gap-stratified seed-split masks | `../examples/<taut>/01_make_split.py` |
| Train Model A + genA | E/F network, then the A-predicted GDV/GAP pool | `../examples/<taut>/02_train_model_A.py` |
| Train Model B | decoupled scaled-NACR (dENACR) head | `../examples/<taut>/03_train_model_B.py` |
| FSSH@MLIP | composed Model A + Model B surface hopping | `../examples/<taut>/04_run_ml_tsh.py` |
| Harvest | guarded, survivor-conditioned populations | `../examples/<taut>/05_harvest.py` |
| FSSH@PYSEQM reference | AM1/CIS reference trajectories | `../scripts/fssh_pyseqm/run_fssh_ref.py` |

Every step has `--help`, takes an explicit `--device {cpu,cuda}`, and reads/writes the local paths you
pass on the command line.

## 2. Environment
See `installation_and_environment.md`. Set `HIPPYNN_USE_CUSTOM_KERNELS=False` (the float64 path). The
FSSH@PYSEQM reference runs on CPU with one trajectory per physical core (`OMP_NUM_THREADS=1`).

## 3. Key run parameters
- **FSSH@MLIP batching.** `--count` chunks the initial-condition set; keep `--count <= 125` on a 40 GB
  GPU (the joint autograd path runs out of memory near 250). Startup is a tiny fraction of a 10 ps run,
  so larger chunks buy little.
- **Timestep / length.** `--timestep 0.05` fs; enol 1 ps is `--nsteps 20000`, keto 10 ps is
  `--nsteps 200000`.
- **Seeds.** The per-initial-condition hop seed is `422 + global_index`; the 5-seed ensemble split seeds
  are fixed in the example configs.

## 4. Using multiple GPUs
Throughput comes from **chunk packing** -- run several `--start`/`--count` slices, one per GPU -- not from
intra-batch parallelism. There is no cross-GPU communication, so a simple per-GPU loop (or the optional
HPC template submitted as a job array) is enough.

## 5. Common failure modes
- **GPU out-of-memory (FSSH@MLIP).** `--count` too large; keep it `<= 125` on a 40 GB GPU.
- **Amplitude NaNs / frozen hopping (FSSH@MLIP).** A gap floor -- the coupling division must use the raw
  gap with no floor (see `fssh_mlip_guide.md`).
- **Batched-eigh crash (FSSH@PYSEQM).** Do not run the reference as a batched GPU job; the near-conical
  CIS eigensolve can fault for the whole batch on some GPUs. Run it on CPU, one trajectory per core.
- **Training out-of-memory.** Lower `batch_size` in the config (the multistate E/F model is the tightest
  fit); the 2-epoch smoke is there to catch this before a long run.

## 6. Monitoring and comparison
Each step prints progress and a completion marker; watch the step's log/stdout. When a run finishes,
harvest it (`../examples/<taut>/05_harvest.py`, or `../scripts/analysis/harvest_fssh_full.py`) and compare
the MLIP and reference populations like-for-like -- same initial conditions, manifold, initial state,
timestep, and survivor conditioning. Report both survivor-conditioned and original-N normalizations.

See also: each stage's guide (`model_a_guide.md`, `model_b_usage_guide.md`, `nve_at_mlip_guide.md`,
`fssh_mlip_guide.md`, `fssh_pyseqm_guide.md`) and `installation_and_environment.md`.
