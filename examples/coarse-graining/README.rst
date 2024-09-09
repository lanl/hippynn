The files in this directory allow one to train and run MD with a coarse-grained HIPNN model. Details of this model can be found in the paper "Thermodynamic Transferability in Coarse-Grained Force Fields using Graph Neural Networks" by Shinkle et. al. available at <https://doi.org/10.48550/arXiv.2406.12112>. 

Before executing these files, one must download the training data from <https://doi.org/10.5281/zenodo.13717306>. The file should be placed at `datasets/cg_methanol_trajectory.npz` where `datasets/` is at the same level as the hippynn repository.

1. Run `cg_training.py` to generate a model. This model will be saved in `hippynn/examples/coarse-graining/model`.
2. Run `cg_md.py` to run MD using the model trained in step 1. The resulting trajectory will be saved in `hippynn/examples/coarse-graining/md_results`.

