# FSSH@PYSEQM per-core worker scripts (vendored)

The reference SLURM scripts in `../` call these per-core dispatch workers via `srun --cpu-bind=cores`.
Each `srun` task runs one worker instance on one physical core; the worker loops over a share of the
initial-condition (IC) indices and calls `run_fssh_ref.py` once per IC (one FSSH trajectory per core).
Vendored here verbatim from the Darwin project root (`$B`) for completeness and provenance.

| worker | called by | IC→core assignment |
|---|---|---|
| `ref_worker_tdc.sh` | `run_ref_6node_hfd.slurm` | static round-robin (`k = SLURM_PROCID; k += SLURM_NTASKS`) |
| `ns3_worker.sh` | `ns3_ref.slurm` | static round-robin |
| `keto_worker_dynamic.sh` | `keto_ns3_10ps.slurm` | dynamic work queue: every core scans all IDs and CLAIMS each via an atomic `mkdir $CLAIMDIR/<idx>` (fast cores absorb the tail; no core stranded near walltime) |
| `ref_progress.py` | monitor (printed by `keto_ns3_10ps.slurm`) | reads the output/log dirs to report completion progress |

All three shell workers:
- read the run parameters from the environment the SLURM script exports (`FORM`, `INIT`, `NSTATES`,
  `NIC`, `START`, `NSTEPS`, `OUTDIR`/`PREFIX_BASE`/`TDC`, and for keto `LOGDIR`/`CLAIMDIR`);
- are **resume-safe**: an IC whose log already contains `REF_DONE` is skipped before any work;
- invoke `$B/run_fssh_ref.py` with `--timestep 0.05 --interval 40 --device cpu` and the coords/velocity
  files at `$B/ics/sampled_${FORM}/{coords.xyz,velocity.out}`.

## Runtime path note
The SLURM scripts reference `$B/<worker>` and `$B/run_fssh_ref.py` at the confirmed Darwin project root
`$B = /vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated`. These vendored copies are
for reference/reproducibility; at runtime the workers, `run_fssh_ref.py`, the `venv_master` interpreter,
and the `ics/sampled_<form>/` inputs are all expected under `$B`. To run from a different root, set `$B`
and place the worker + `run_fssh_ref.py` (see `../run_fssh_ref.py`) + ICs there.

Provenance: pulled 2026-08-30 from `$B/` (4 files). The three shell workers carry the project
`PURPOSE/INPUTS/OUTPUTS/CONFIG/DEPENDS` header block plus the dev-header line; the refactor is
additive (header and comments only) and behavior-preserving — every executable line is byte-identical
to the vendored original.
