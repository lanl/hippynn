The files in this directory illustrate how to train and run MD with a coarse-grained HIPNN model. The scripts
``cg_training.py`` and ``cg_md.py`` align with the work from the paper "Thermodynamic Transferability in Coarse-Grained
Force Fields using Graph Neural Networks" by Shinkle et al., available at <https://doi.org/10.48550/arXiv.2406.12112>.
The script ``des_cg_example.py`` illustrates how to extend the workflow to deep eutectic solvents (DESs), which involve
multiple components/bead types.

### Methanol example

To use the ``cg_training.py`` and ``cg_md.py`` scripts, one must download the training data from
<https://doi.org/10.5281/zenodo.13717306>. The file should be placed at ``datasets/cg_methanol_trajectory.npz``, where
``datasets/`` is at the same level as the hippynn repository.

Alternatively, one can follow the steps in the notebook ``aa_to_cg_workflow.ipynb`` to generate their own training data
from LAMMPS or Gromacs output files. However, it is suggested to try the provided dataset first.

1. Training the model:
    Run ``cg_training.py`` to generate a model. This model will be saved in ``hippynn/examples/coarse-graining/model``.
    If you have a LAMMPS installation with the ML-IAP package and Python bindings, a LAMMPS unified potential file will
    also be saved in the directory. This unified potential file can also be generated at a later time using the script
    ``save_model_for_lammps.py``. If you don't intend to use the potential with LAMMPS, you can ignore this.
2. Running MD with the model:
    1. With the built-in hippynn MD driver:
        Run ``cg_md.py``. The resulting trajectory will be saved in ``hippynn/examples/coarse-graining/md_results``.
    2. With LAMMPS:
        You will need a LAMMPS installation with the ML-IAP package and Python bindings. Before running LAMMPS,
        execute ``write_lammps_data_file.py`` to create a LAMMPS data file from the last frame of the training data
        file. This will be saved in the directory ``lammps_md_inputs``. There is already an example input script in
        that directory. Move to the directory and run LAMMPS with the provided input file.

**Caution**: RepulsivePotentialNode in ``repulsive_potential.py`` has been modified. Any CG model trained before this
change (3/21/25) will not be compatible with these updates. Please train a new model. Results should not be impacted in
any way, besides potential effects of uncontrolled random variables.

### Deep eutectic solvent (DES) example

One should first download the files for the desired configuration from XXXX. They should be placed in
``datasets/des_cg/``, where ``datasets/`` is at the same level as the hippynn repository.

``des_cg_example.py`` trains a HIP-HOP-NN CG model for ethaline, reline, or hydrated reline, runs CG MD with the hippynn
MD driver, and compares the CG partial RDFs to the all-atom reference and to the CG trajectory from the paper. Choose
the system, temperature, and mapping (COM, COG, or Martini for reline) at the top of the script. For the Martini mapping,
which splits choline into two beads, the model is trained on all four bead types and the two choline beads are merged
back into one before the RDFs are computed. Training takes 30-60 minutes on a GPU and the paper-length MD takes
several hours; the stage flags and step counts at the top of the script can be used to rerun individual stages or to
shorten the run. The polynomial repulsive potential it uses is ``RepulsivePotentialPolynomialBySpeciesNode`` in
``repulsive_potential.py``.
