Collaborators: Ajay Khanna, Tomas Karella both postdoc at LANL
Objectives:
a). Inference optimization for all predictions, especially FSSH driven by MLIPs
b). Merging the MLIP for NAMD new features to HIPPYNN and update it

1. Fork and convert current HIPPYNN repo (https://github.com/lanl/hippynn, developer branch) to private with branch name "ml_namd" in my Github repo, if we can rename it. Clone the repo afterwards to this folder: "C:\Users\401770\OneDrive - Los Alamos National Laboratory\PostDoc\LANL\ML_Projects\Codes\hippynn_namd_project"
2. Currently working HIPPYNN conda/micromamba environment file/requirement.txt for exact replication of our environment. Similar for the PySEQM (https://github.com/lanl/PYSEQM, branch: master) venv_master environment requirement.txt
#------------------------------------------------------
Reproducible environments

Create an exact reproducibility record for:

The currently working HIPPYNN Conda or micromamba environment.
The currently working PYSEQM environment named venv_master, if that is its exact name.

Do not assume that requirements.txt alone can reproduce CUDA-enabled scientific environments.

Determine and record:

operating system
CPU architecture
Python version
environment manager and version
PyTorch version
CUDA runtime and toolkit version
GPU model and driver version
HIPPYNN commit
PYSEQM commit
NumPy and other compiled dependency versions
pip-installed packages
Conda/micromamba-installed packages
editable local packages
relevant environment variables
required external libraries
SLURM and HPC module dependencies

Propose the appropriate reproducibility artifacts, which may include:

a human-readable environment.yml
a platform-specific explicit Conda or micromamba lock/export
a pip requirements.txt for pip-managed packages
environment creation and activation instructions
an environment verification script

Do not replace an existing environment file until the generated environment has been tested.

#------------------------------------------------------
3a) Dataset (local or cloud): S0-dynamics dataset for MODEL-A training and enol-form's BALF (clean geometries) dataset for model B training. Vishikhs's 1000-IC enol-form configurations for reference and mlip FSSH dynamics.
3b) Dataset extraction/generation scripts from PySEQM
4. Model A training script [E and F only] (S0-S5 PES)
5. Model B training script [dENACR] (3 Pairs: S1-S2, S1-S3, and S2-S3), v3 version that takes Model A's energy gaps and gradient difference vectors at inference
6. Model A-based NVE@MLIP dynamics script
7. Model A: inference + Model B-based FSSH@MLIP script, the phase_sign vectorized, script
8. FSSH@PySEQM scripts
9. SLURM scripts for running ALL jobs. 

First copy these latest script we have been using to this folder:"C:\Users\401770\OneDrive - Los Alamos National Laboratory\PostDoc\LANL\ML_Projects\Codes\hippynn_namd_project". Then we need to the following to all python and slurm scripts:
1. Rewrite them will full, no-assumption, doc-string and self-explanatory and intuititve constants, variable, function, loops, inputs, and output names. The script should be rewritten in a way that makes them really simple to read and self-explanatory. This rewrite should not affect their ability to the jobs they are built for.
2. Insert developer information in all of them:"
# Developer Information
---------------
  __developer__: "Ajay Khanna"
  __place__: "LANL/Sergei Lab"
  __date__: "Sep-01-2026"
  __project__: "ml_namd"
"
3. Model B v3-inference-updated script should have two things: a markdown file showing the mathematical background used to built the dENACR learning/training path and a detailed how-to-use markdown file
#---------------------------
Required documentation
At minimum, produce or update:

top-level README.md
installation and environment instructions
repository and upstream provenance
data manifest and schema documentation
Model A training and inference guide
Model B mathematical background
Model B usage guide
NVE@MLIP guide
FSSH@MLIP guide
FSSH@PYSEQM guide
SLURM job guide
validation report
inference benchmark report
migration/refactoring change log
known limitations and unresolved questions

#---------------------------
Refactoring standard

Refactor every project-owned Python and SLURM script for clarity, maintainability, and reproducibility while preserving its scientific and numerical behavior.

Python requirements

Each Python file must have:

a complete module docstring
a concise statement of scientific purpose
documented inputs and outputs
documented units and tensor shapes
descriptive constants, variables, functions, classes, and loop indices
type hints where practical
function docstrings describing parameters, returns, units, shapes, side effects, and exceptions
an explicit main() function where appropriate
an if __name__ == "__main__": guard for executable scripts
structured configuration rather than scattered unexplained literals
explicit device and dtype handling
clear validation and actionable error messages
controlled logging instead of unexplained prints
deterministic random-seed handling where relevant
safe path handling
no embedded credentials
no undocumented absolute machine-specific paths
comments explaining scientific intent rather than restating syntax

Do not rename established scientific terms merely to make them shorter. When a concise symbol is conventional, define it clearly at its first use.

SLURM requirements

Each SLURM script must have:

a purpose header
valid shebang and #SBATCH directives
documented required inputs
documented outputs and logs
clearly labeled user-configurable settings
explicit environment activation
defensive shell behavior where compatible
directory and file existence checks
quoted paths
clear launch command
failure reporting
no secrets
no unexplained hard-coded machine paths

All example commands must use placeholders for machine-specific paths unless the command is explicitly intended for the confirmed environment.
#---------------------------


Clarify all question you need to thoroughly understand this massive task. Currently your goal is to copy the fork and close the hippynn repo, aforementioned files and rewritting only the files we own


