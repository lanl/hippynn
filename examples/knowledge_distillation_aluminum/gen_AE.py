""" 
    Example scripts for 
    Teacher-student training improves accuracy and efficiency of machine learning interatomic potentials
    https://arxiv.org/abs/2502.05379

    This is an example script to generate atomic energies using a trained 
    teacher model 
    (example: https://github.com/lanl/hippynn/blob/development/examples/ani_aluminum_example.py),
    and is designed for an external dataset available at
    https://github.com/atomistic-ml/ani-al. 

    Note: It is necessary to untar the h5 data files before running this script.

    For information on the dataset, see publication:
    Smith, J.S., Nebgen, B., Mathew, N. et al.
    Automated discovery of a robust interatomic potential for aluminum.
    Nat Commun 12, 1257 (2021).
    https://doi.org/10.1038/s41467-021-21376-0

"""
import os
import numpy as np
import argparse
import torch 

import hippynn
from hippynn.tools import active_directory
from hippynn.experiment.serialization import load_checkpoint_from_cwd
import hippynn.graphs
hippynn.settings.WARN_LOW_DISTANCES = False


def main(args):
    """Generate database with auxiliary targets. 

    Args:
        args : Arguments from command line via arg parser. 

    Raises:
        FileNotFoundError: _description_
    """
    if args.use_gpu:
        torch.cuda.set_device(args.gpu)
    
    try:
        with active_directory(args.model_loc, create=False):
            bundle = load_checkpoint_from_cwd(map_location="cpu", restore_db=False)  
    except FileNotFoundError:
        raise FileNotFoundError("Model not found!")
    
    # Add the atom-energies to the predictor
    model = bundle["training_modules"].model
    h_energy = model.node_from_name("HEnergy")
    atomic_energy = h_energy.atom_energies
    model.nodes_to_compute.append(atomic_energy)
    predictor = hippynn.graphs.Predictor.from_graph(model)

    # Dataset Preprocessing (similar to teacher model). 
    from hippynn.databases.h5_pyanitools import PyAniDirectoryDB
    
    database = PyAniDirectoryDB(
        directory=args.data_loc,
        seed=None, 
        quiet=False,
        allow_unfound=True,  
        inputs=None,
        targets=None,
    )

    import ase
    energy_shift = 72  # eV
    arrays = database.arr_dict
    
    R = torch.from_numpy(arrays["coordinates"])
    Z = torch.from_numpy(arrays["species"]) 
    cell = torch.from_numpy(arrays["cell"])
    n_atoms = arrays["species"].bool().int().sum(dim=1)
    F = arrays["force"] * (ase.units.Hartree / ase.units.eV)
    T = arrays["energy"] * (ase.units.Hartree / ase.units.eV) + energy_shift * n_atoms

    # Use Predictor to compute atomic energies (AE).  
    outputs = predictor(
        species=Z, 
        coordinates=R,
        cell=cell, 
        batch_size=256,
    ) 
    AE = outputs["HEnergy.atom_energies"]
    AE = torch.squeeze(AE)
        
    # Save all arrays as numpy arrays. 
    fname = "data-from-teacher_Al"
    if not os.path.exists(args.aug_target_loc):
        os.makedir(args.aug_target_loc)
    np.save(f"{args.aug_target_loc}/{fname}_Z.npy", Z.cpu().detach().numpy())
    np.save(f"{args.aug_target_loc}/{fname}_R.npy", R.cpu().detach().numpy())
    np.save(f"{args.aug_target_loc}/{fname}_cell.npy", cell.cpu())
    np.save(f"{args.aug_target_loc}/{fname}_T.npy", T)
    np.save(f"{args.aug_target_loc}/{fname}_F.npy", F)
    np.save(f"{args.aug_target_loc}/{fname}_AE.npy", AE.cpu().detach().numpy())
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    from argparse import BooleanOptionalAction
    
    parser.add_argument("--teacher_loc", type=str, help="Location of Teacher model")
    parser.add_argument("--data_loc", type=str,help="Location of training data")
    parser.add_argument("--aug_target_loc", type="str", help="Folder for augmented dataset", nargs="?", default="test_data")
    parser.add_argument(
        "--use-gpu",
        action=BooleanOptionalAction,
        default=torch.cuda.is_available(),
        help="Whether to use GPU. Defaults to torch.cuda.is_available()",
    )
    
    args = parser.parse_args()
    
    main(args)
    
