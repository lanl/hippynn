# Inference benchmark report

Throughput and memory measurements for MLIP inference and the FSSH@MLIP driver. This is the evidence
base for objective (a), inference optimization — especially the FSSH path. Numbers are representative
A100 measurements from the campaign; re-measure on your hardware before quoting.

---

## 1. Model A inference throughput (A100)

| Quantity | Rate | Per geometry | Notes |
|---|---|---|---|
| Energies only (`E_k`) | ~3.3–3.6k geom/s | ~0.28 ms | forward pass only |
| Energies + forces (`E_k, F_k`) | ~405 geom/s | ~2.46 ms | forces add an autograd pass, ~9× the energy cost |
| Gaps `E_j − E_i`, GDV `F_i − F_j` | free | — | subtractions of already-computed quantities |

Model A carries ~2.6M parameters. The dominant inference cost when forces are required (FSSH and NVE
both need them) is the force autodifferentiation — roughly 9× the energy-only forward. Because the gaps
and gradient-difference vectors Model B consumes are simple subtractions of Model A outputs, they add no
measurable cost; the scaled coupling itself (dENACR) is a separate Model B evaluation, not a Model A
output.

**Implication:** plan any force-bearing inference (FSSH, NVE) around the ~2.5 ms/geometry figure, and
avoid recomputing forces where a cached value suffices.

## 2. FSSH@MLIP batching and memory

The FSSH@MLIP driver propagates a batch of independent trajectories on one GPU. Measured on a 40 GB
A100:

- **`COUNT = 125` initial conditions per chunk is the OOM-safe maximum.** The joint autograd path OOMs
  at `COUNT = 250`; a `COUNT = 500` run peaks at ~37–39 GB (autograd buffers). Never default `COUNT` above
  125 on 40 GB.
- **Startup is ~10 s**, about 0.2% of a 10 ps run, so larger batches buy essentially nothing beyond the
  memory ceiling — there is no throughput reason to push `COUNT` higher.

Unlike FSSH@PYSEQM, the MLIP driver has **no CIS eigensolve**, so batched GPU propagation is stable and
fast (no near-CI cuSOLVER faults). This is the structural reason FSSH@MLIP can batch where the reference
cannot.

## 3. Multi-GPU: pack chunks, do not parallelize the batch

Speed across a multi-GPU node comes from **packing independent chunks**, not from splitting one batch:

- Chunk packing (fanning `--start`/`--count` slices across the node's GPUs) gives ~3.7× on 4 GPUs.
- Data-parallel splitting of a single batch (DDP-style) gives only ~1.21×.

A multi-GPU run packs chunks across the node's GPUs with a **dynamic work-queue**: each GPU pulls the next
chunk when it finishes, so heterogeneous chunk runtimes do not strand a fast GPU waiting on a slow one.
Prefer this over static round-robin assignment (see `../optional_hpc/` for a scheduler template).

## 4. FSSH@PYSEQM reference cost (for context)

The reference is CPU, one trajectory per physical core (it cannot batch on GPU; see
`fssh_pyseqm_guide.md`). Representative per-step cost is ~0.30 s/step on AMD Rome, about 1.9× slower than
on Intel SPR; a keto 10 ps reference trajectory takes on the order of ~16 h of core time. A full 1000-IC
reference sweep is run across hundreds of cores (e.g. 6–8 nodes × 112 cores) for a day or two. FSSH@MLIP,
by batching on GPU, produces the equivalent ensemble far faster — the practical motivation for the MLIP
surrogate.

## 5. What to optimize next (objective a)

- **Force reuse.** Forces are the ~9× cost; any place a step can reuse a cached force instead of
  recomputing it is the highest-leverage saving.
- **Batch to the memory ceiling, not beyond.** `COUNT = 125` on 40 GB (proportionally higher on 80 GB);
  larger buys nothing once startup is amortized.
- **Pack, don't DDP.** Chunk packing (3.7×) dominates data-parallel batch splitting (1.21×); the dynamic
  work-queue removes the tail-idle of static assignment.
- **Precision.** Inference runs in float64 with custom kernels disabled (the graph is float64 for
  energy-conserving MD); a validated mixed/float32 inference path for the coupling could raise throughput
  if it does not perturb the near-CI dynamics — untested, a candidate.

## 6. How to reproduce

Model A throughput: time a fixed batch of geometries through the pruned `[Z,R]→[sE_k,F_k]` graph with
and without the force output. FSSH batching: run `../examples/<tautomer>/04_run_ml_tsh.py` at increasing
`--count` and read peak memory from the logs; the OOM boundary is the ceiling. Multi-GPU:
compare wall time for the same total ICs run as one big chunk vs packed chunks under the dynamic queue.

See also: `validation_report.md` (accuracy), `fssh_mlip_guide.md` (the driver + batching), and
`slurm_job_guide.md` (running the pipeline).
