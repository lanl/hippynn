# FSSH@PYSEQM guide: the semiempirical AM1/CIS surface-hopping reference

FSSH@PYSEQM runs the same fewest-switches surface hopping as FSSH@MLIP, but with PySEQM's native
AM1/CIS electronic structure computing energies, forces, and couplings on the fly. It is the
**reference** against which FSSH@MLIP is measured: identical dynamics engine, identical initial
conditions, only the electronic-structure source differs. The apple-to-apple population comparison lives
here.

Scripts live in `../scripts/fssh_pyseqm/` (drivers, harvest utilities) and
`../scripts/fssh_pyseqm/workers/` (per-core dispatch). The MLIP counterpart is `fssh_mlip_guide.md`.

Placeholders: `$B` = the reference project root on the cluster (holds the driver, `venv_master`, and the
`ics/` inputs); replace with your own.

---

## 1. Role and method

One trajectory per initial condition, with:

- **AM1/CIS** electronic structure, `n_states` excited states;
- **`tdc_method = hamiltonian_fd`** (finite-difference overlap time-derivative coupling) — the accurate
  TDC available to a method that *has* CIS amplitudes (unlike the MLIP, which uses `nac_dot_v`);
- **`decohere_on_hop = True`** (project rule; without decoherence the quantum populations stay trapped
  on the initial surface);
- **microcanonical** propagation with **GS-sampled velocities** injected from the developer initial
  conditions (Vishikh convention: raw `velocity.out` divided by 1000);
- **CPU** execution, one trajectory per physical core (Section 2).

The driver is `run_fssh_ref.py`; `run_fssh_ref_dc.py` is the same with a `--detect-crossings {true,false}`
A/B knob for the trivial-crossing detector.

## 2. Why CPU, one trajectory per core

FSSH@PYSEQM is deliberately **not** run as a batched GPU job. Near a conical intersection the CIS
eigenproblem becomes ill-conditioned; the batched cuSOLVER `eigh` on an A100 then raises for the
**entire batch** (a fault-isolation artifact, not a physics or subspace problem). The same seed survives
on CPU (LAPACK) and on Volta GPUs; it crashes only on the A100 batched path. The fix, adopted here, is
CPU execution with **one FSSH trajectory per physical core** — LAPACK is robust, cores are independent,
and a single bad trajectory cannot take down its neighbors. Threading is pinned to one thread per
trajectory (`OMP_NUM_THREADS=1`, etc.) to avoid oversubscription.

## 3. The driver and its inputs

`run_fssh_ref.py` reads one frame of a multi-frame `.xyz` (coordinates) and the matching frame of a
NEXMD `velocity.out`, builds a PySEQM molecule + `SurfaceHoppingDynamics`, and runs. Key CLI:

```
--coords coords.xyz      multi-frame xyz (developer ICs)
--velocities velocity.out NEXMD velocity file (frames delimited by $VELOC)
--idx N                  which IC frame to run
--n-states K             CIS excited states (S0..SK)
--initial-state S        starting surface (e.g. 3)
--nsteps                 steps (10 ps at dt=0.05 fs = 200000)
--timestep 0.05          fs
--interval 40            store every 40 steps (= 2 fs)
--tdc hamiltonian_fd     reference TDC
--out-dir / --prefix     output location; --device cpu
```

Two input conventions are load-bearing:

- **Velocity scaling.** The raw `velocity.out` values are divided by 1000 to reach PySEQM units
  (`VEL_DIV = 1000`).
- **Descending-Z ordering.** The master PySEQM API requires species non-increasing per row, so the
  driver sorts atoms by **decreasing atomic number** and permutes R and V with the same order. For
  `acn_Z = [8,8,6,6,6,6,6,1×8]` this is already sorted, but the permutation is applied stably regardless
  — do not assume it is a no-op for a differently-ordered file.

The seed is `422 + idx`, matching FSSH@MLIP so the two share initial conditions per index.

## 4. Running it (workers + SLURM)

`srun` launches one worker per physical core; each worker loops over a share of the IC indices and calls
`run_fssh_ref.py` per IC. Workers live in `../scripts/fssh_pyseqm/workers/`:

- `ref_worker_tdc.sh` — static round-robin (`k = SLURM_PROCID; k += SLURM_NTASKS`).
- `keto_worker_dynamic.sh` — a **dynamic cross-node work queue**: every core scans all IDs and claims
  each via an atomic `mkdir $CLAIMDIR/<idx>` (atomic on the shared filesystem). Fast cores absorb the
  tail so no core is stranded with two slow trajectories near the walltime.

All workers are **resume-safe**: an IC whose log already contains `REF_DONE` is skipped before any work.

SLURM launchers (`../scripts/fssh_pyseqm/`):

- `run_ref_6node_hfd.slurm` — enol init3, `n_states=4`, 1000 Vishikh ICs, 6 nodes × 112 cores.
- `ns3_ref.slurm` — enol n3 (S0–S3) 100-IC apple-to-apple partner, 1 node, debug qos.
- `keto_ns3_10ps.slurm` — keto n3 10 ps, 1000 ICs, 8 nodes × 112 cores, dynamic queue, one job.

## 5. Failures are recorded, not raised

A large IC sweep must not abort on one bad IC, so the driver is **skip-not-raise**: any exception is
caught, written into the per-trajectory `<prefix>.summary.json` with a `failed` status and the error
string, and the process exits non-zero for that IC only. The dominant failure is a near-CI eigensolve
`LinAlgError`; for the keto 10 ps reference ~225/1000 ICs die this way, ~94% of them at the S1–S0 CI as
the cascade completes into the ground-state funnel. These deaths are physical trajectories that reached
an ill-conditioned seam, not a code bug — they are excluded from survivor-conditional ensembles.

## 6. Outputs and harvest

Each trajectory writes a PySEQM h5 with `data/nonadiabatic/{active_surface, NACT, electronic_amplitudes}`
and `data/excitation/state_energies`. Harvest utilities (`../scripts/fssh_pyseqm/`):

- `harvest_fssh_full.py` — the canonical 6-panel apple-to-apple harvest (survivor-conditional live
  ensemble; classical = active-surface fraction, quantum = excited $|c|^2$).
- `survivors_ensemble.py` — survivors-only with a common fixed N.
- `fixedN_harvest.py` — fixed-N (total conserved, ground-state recovery visible).
- `harvest_clean.py` — strict clean survivors (excludes any NaN-amplitude trajectory).

Use the **same harvester and conventions** for the reference and the MLIP so the two `.npz` files are
directly comparable.

## 7. Apple-to-apple matching rules

For a valid FSSH@MLIP-vs-FSSH@PYSEQM comparison, match:

- **Initial conditions** — same coords/velocity frames, same per-index seed (`422 + idx`).
- **Manifold** — same states and coupling pairs (e.g. S0–S3, pairs 12/13/23).
- **Initial surface, timestep, store interval.**
- **Survivor conditioning** — harvest both with the same normalization.

One difference is intrinsic and cannot be matched: the reference TDC is `hamiltonian_fd` while the MLIP
uses `nac_dot_v` (the MLIP has no CIS amplitudes). This is validated harmless at the ensemble level;
note it when reporting. Survival asymmetry is also expected — the reference loses trajectories to eigh
deaths that the MLIP (no CIS eigensolve) does not — so compare survivor-conditional populations, and
account for the differing survivor counts.

## 8. Gotchas

- **Descending-Z + velocity/1000** are both required; getting either wrong silently corrupts the
  dynamics.
- **CPU only** — do not move this to a batched GPU path (Section 2).
- **`hamiltonian_fd` + decohere** define the reference; changing the TDC or turning off decoherence makes
  it a different reference (and no longer comparable to the established runs).
- **Resume** wipes the claim directory but keeps `REF_DONE` skips, so a restart re-runs only unfinished
  trajectories.

## 9. Verify

1. Environment: the Darwin `venv_master` (master PySEQM); see `../env/`.
2. Small pilot: run 5–10 ICs on one node (`ns3_ref.slurm` style) and confirm each writes an h5 + a
   `summary.json` with `status: ok` and a nonzero hop count.
3. Harvest the pilot with `harvest_fssh_full.py` and sanity-check the initial populations (all on the
   initial surface at t=0).

See also: `fssh_mlip_guide.md` (the MLIP driver this benchmarks), `model_b_math_background.md` (the
coupling physics), and `slurm_job_guide.md`.
