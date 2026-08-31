# SLURM job guide: submitting and monitoring the pipeline

How to run every stage of the pipeline as a cluster job: training, data generation, NVE@MLIP,
FSSH@MLIP, and FSSH@PYSEQM. Each stage's science is in its own guide; this document is the operational
layer — clusters, environments, submission, parameters, monitoring, and recovery. All example paths are
the confirmed cluster locations; replace them with your own.

---

## 1. Job families at a glance

| Stage | Script(s) | Cluster | Environment | Resource |
|---|---|---|---|---|
| Train Model A / B | `../scripts/training/train_model_{a,b}_enol.slurm` | Chicoma | conda `hiphop_env` | 1 GPU (gpu40) |
| Data generation | `../scripts/data_generation/bg_gen_enol.slurm`, `bg_ext_enol.slurm`, `galf_gen_enol.slurm` | Chicoma | conda `mlip_md_dyn` | 4 GPU / 64 CPU |
| NVE@MLIP | `../scripts/nve_mlip/nve_smoke_dt01.slurm`, `nve_ms_pack.slurm`, `nve_prod_chicoma.slurm` | Chicoma | conda `mlip_md_dyn` | 1–4 GPU |
| FSSH@MLIP | `../scripts/slurm/keto_mlip_prod.slurm` (+ `_s0s5A`, `_s0s3_jointB`, and `_smoke`) | Chicoma | `venv_master` | 4 GPU (dynamic queue) |
| FSSH@PYSEQM | `../scripts/fssh_pyseqm/{run_ref_6node_hfd,ns3_ref,keto_ns3_10ps}.slurm` | Darwin | `venv_master` | many CPU cores |

## 2. Cluster cheat-sheet

**Chicoma** (GPU work). `--account=s17_cint_g`, `--partition=gpu` (production) or `gpu_debug` (short,
reservation), `--qos=long`/`debug`, `--constraint=gpu40` (or `gpu80`). A100 40 GB or 80 GB nodes, 4
GPUs/node. Standard-queue walltime is capped (use `qos=long` for multi-day). Login nodes have no GPU
(`nvidia-smi` fails there).

**Darwin** (CPU FSSH@PYSEQM). `--account=y2020-bf`, `--partition=shared-spr`, `--qos=long`/`debug`,
112 tasks/node × `--cpus-per-task=2`, `--exclusive`. One FSSH trajectory per physical core.

## 3. Environment activation

Each script activates its own environment; keep those lines pointing at your install
(`installation_and_environment.md`):

- **Training** and **NVE / data-gen** slurms `source` a conda profile and `conda activate` an env
  (`hiphop_env` for training, `mlip_md_dyn` for NVE/data-gen).
- **FSSH@MLIP** and **FSSH@PYSEQM** call the `venv_master` interpreter directly.

## 4. Submitting each stage

**Training.**

```bash
sbatch ../scripts/training/train_model_a_enol.slurm          # Model A (E+F, S0-S5), full run
SMOKE=1 sbatch ../scripts/training/train_model_b_enol.slurm   # Model B (dENACR), 2-epoch smoke
```

The `.inp` in `../scripts/training/configs/` is self-contained (data_dir/output_dir inside); `SMOKE=1`
overrides to 2 epochs. Output: a model directory with `experiment_structure.pt`, `best_model.pt`,
`training_summary.json`, and parity CSVs.

**Data generation.**

```bash
sbatch ../scripts/data_generation/bg_gen_enol.slurm   # PySEQM AM1/CIS labels for enol BALF+GALF
sbatch ../scripts/data_generation/bg_ext_enol.slurm   # extract -> acn_ splits (after gen)
```

Needs the PySEQM-ml_namd fork on `PYTHONPATH` (set inside the script). Sentinels: `BG_GEN_DONE`,
`BG_EXT_DONE`.

**NVE@MLIP.**

```bash
sbatch ../scripts/nve_mlip/nve_smoke_dt01.slurm            # dt-convergence smoke
sbatch ../scripts/nve_mlip/nve_ms_pack.slurm              # optimize S1-S3 then NVE S0-S3 x {enol,keto}
TAUT=enol sbatch ../scripts/nve_mlip/nve_prod_chicoma.slurm  # 100 ps production grid (submit twice)
```

**FSSH@MLIP.** Parameterized via `--export`:

```bash
sbatch --export=ALL,MODE=comp,CELL=BALF,COUNT=125,NIC_TOTAL=1000 \
       ../scripts/slurm/keto_mlip_prod.slurm
```

`MODE={joint|comp}`, `CELL={BALF|GALF}`, `COUNT` = ICs per chunk (≤ 125 on 40 GB), `NIC_TOTAL` = total
ICs. The script runs a dynamic FIFO GPU work-queue over the node's 4 GPUs. Smoke first with
`../scripts/slurm/keto_mlip_smoke.slurm` (or `keto_s0s3_jointB_smoke.slurm`).

**FSSH@PYSEQM (Darwin).**

```bash
sbatch ../scripts/fssh_pyseqm/keto_ns3_10ps.slurm    # keto n3 10 ps, 1000 ICs, dynamic queue
FORM=enol INIT=3 NSTATES=4 sbatch ../scripts/fssh_pyseqm/run_ref_6node_hfd.slurm
```

The workers (`../scripts/fssh_pyseqm/workers/`) fan ICs across cores; the keto worker uses an atomic
claim queue.

## 5. Monitoring

- **Queue / history.**

  ```bash
  # Chicoma
  ~/bin/chfe 'squeue -u <user> -o "%.10i %.16j %.9T %.10M %.9L %R"'
  ~/bin/chfe 'sacct -u <user> -S today -X -o JobID,JobName%16,State,Elapsed,End'
  # Darwin
  ssh <user>@darwin-fe.lanl.gov 'squeue -u <user>'
  ```

  (On Windows the Chicoma helper needs the `MSYS_NO_PATHCONV=1` prefix for absolute-path remote commands.)

- **Completion sentinels.** Each script prints a distinctive `*_DONE` line on success
  (`TRAIN_A_ENOL_DONE`, `MULTISTATE_DONE`, `KETO_NS3_10PS_DONE`, `D1_COMPOSED_DONE`, `REF6_DONE`, ...);
  grep the logs for these plus the per-IC `REF_DONE` markers.
- **FSSH@PYSEQM progress.** `ref_progress.py` reads the output/log dirs and reports completion; the
  keto slurm prints the exact monitor command at launch.
- **Harvest** when a run finishes: `../scripts/analysis/harvest_fssh_full.py` (FSSH@MLIP) or the
  `../scripts/fssh_pyseqm/harvest_*.py` utilities (reference), then compare apple-to-apple.

Note (project convention): do not poll cluster jobs in a busy loop; stage the harvest and wait for the
run to finish.

## 6. Resume, chunking, and multi-GPU

- **Resume-safe.** FSSH@PYSEQM workers skip any IC whose log already contains `REF_DONE`; a restart runs
  only unfinished trajectories (the keto slurm wipes its claim directory first). Training jobs restart
  from the config; use a fresh output dir or the restart flags.
- **Chunking.** FSSH sweeps are chunked with `--start`/`--count` (MLIP) or IC ranges (reference). Keep
  FSSH@MLIP `COUNT ≤ 125` on 40 GB.
- **Multi-GPU.** Speed comes from packing chunks across GPUs, not intra-batch parallelism. The FSSH@MLIP
  prod slurm's dynamic FIFO queue is preferred over static round-robin, which can strand a GPU on a slow
  chunk.

## 7. Common failure modes

- **GPU OOM (FSSH@MLIP).** `COUNT` too large; keep ≤ 125 on 40 GB (the joint autograd path OOMs at 250).
- **Dead-GPU cascade.** Manual `CUDA_VISIBLE_DEVICES` packing of several processes into one job can lose
  a GPU and cascade. Prefer one clean `gres=gpu:1` per job (e.g. a job array), or the dynamic queue.
- **Batched-eigh crash (FSSH@PYSEQM).** Do not run the reference as a batched GPU job; the near-CI
  cuSOLVER `eigh` faults for the whole batch on A100. Use CPU one-per-core.
- **Amplitude NaNs / frozen hopping (FSSH@MLIP).** A gap floor; the coupling division must use the raw
  gap (see `fssh_mlip_guide.md`).
- **Training OOM at batch 64 on 40 GB.** Lower the batch in the `.inp` (the multistate 6-head E+F is the
  tightest fit); the smoke exists to catch this before a long run.

See also each stage's guide (`model_a_guide.md`, `model_b_usage_guide.md`, `nve_at_mlip_guide.md`,
`fssh_mlip_guide.md`, `fssh_pyseqm_guide.md`) and `installation_and_environment.md`.
