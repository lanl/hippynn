The files in this directory allow one to train and run MD with a coarse-grained HIPNN model. Details of this 
model can be found in the paper "Thermodynamic Transferability in Coarse-Grained Force Fields using Graph Neural 
Networks" by Shinkle et. al. available at <https://doi.org/10.48550/arXiv.2406.12112>. 

Before executing these files, one must download the training data from <https://doi.org/10.5281/zenodo.13717306>. 
The file should be placed at ``datasets/cg_methanol_trajectory.npz`` where ``datasets/`` is at the same level as the 
hippynn repository.

Alternatively, one can follow the steps in the notebook ``aa_to_cg_workflow.ipynb`` for generate their own training 
data from LAMMPS or Gromacs output files. However, it is suggested to try with the provided dataset first.

1. Training the model: 
    Run ``cg_training.py`` to generate a model. This model will be saved in 
    ``hippynn/examples/coarse-graining/model``. If you have a LAMMPS installation with the ML-IAP package and python 
    bindings, a LAMMPS unified potential file will also be saved in the directory. This unified potential file can 
    also be generated at a later time using the script ``save_model_for_lammps.py``. If you don't care to use the 
    potential with LAMMPS, you do not need to mind any of this. 
2. Running MD with the model:
    1. With the built-in hippynn MD driver: 
        Run ``cg_md.py``. The resulting trajectory will be saved in 
        ``hippynn/examples/coarse-graining/md_results``.
    2. With LAMMPS: 
        You will need to have a LAMMPS installation with the ML-IAP package and python bindings. 
        Before running LAMMPS, execute the ``write_lammps_data_file.py`` to create a LAMMPS data file from the last 
        frame of the training data file. This will be saved in the directory ``lammps_md_inputs``. There is already 
        an example input script in that directory. Move to the directory and run LAMMPS with the provided input file.  

**Caution**: RepulsivePotentialNode in ``repulsive_potential.py`` has been modified. Any CG model trained before this 
change (3/21/25) will not be compatible with these updates. Please train a new model. Results should not be impacted 
in any way, besides potential effects of uncontrolled random variables. 