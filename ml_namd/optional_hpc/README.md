# optional_hpc — generic batch-scheduler template (not required)

The ml_namd pipeline does **not** require a job scheduler. Every step in
`ml_namd/examples/<tautomer>/` runs as a plain `python <step>.py ...` on any machine that has
a CUDA GPU and the project environment installed.

`run_on_hpc.slurm.template` is only a convenience for users who prefer to submit steps to a
Slurm cluster. It is intentionally generic: replace every `<PLACEHOLDER>` with your own
account, partition, and walltime. Nothing in this folder is tied to any specific cluster,
filesystem, or account.
