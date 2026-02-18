"""
Pytorch lightning example script for training HIP-NN using data split across multiple ranks.

See also:
 split_ani1x.py (splits the data, must be run first)
 job_ani1x.py (simple slurm script that runs the python files)

This script was designed for an external dataset available at
https://doi.org/10.6084/m9.figshare.c.4712477

For info on the dataset, see the following publication:
Smith, J.S., Zubatyuk, R., Nebgen, B. et al.
The ANI-1ccx and ANI-1x data sets, coupled-cluster and density functional
theory properties for molecules. Sci Data 7, 134 (2020).
https://doi.org/10.1038/s41597-020-0473-z

"""

import argparse
import torch
import hippynn
import ase.units


def make_model(network_params, tensor_model, tensor_order, tensor_factors, atomization_consistent):
    """
    Build the model graph for energy and potentially force prediction.
    """
    from hippynn.graphs import inputs, networks, targets, physics

    if tensor_order == 0 or tensor_model == "NONE":
        net_class = networks.Hipnn
    else:
        if tensor_model == "TS":
            net_class = {
                1: networks.HipnnVec,
                2: networks.HipnnQuad,
            }[tensor_order]
        elif tensor_model == "HOP":
            net_class = networks.HipHopnn
            network_params["l_max"] = tensor_order
            network_params["n_max"] = tensor_factors

    species = inputs.SpeciesNode(db_name="atomic_numbers")
    positions = inputs.PositionsNode(db_name="coordinates")
    network = net_class("hipnn_model", (species, positions), module_kwargs=network_params)

    if not atomization_consistent:
        henergy = targets.HEnergyNode("HEnergy", network)
    else:
        henergy = targets.AtomizationEnergyNode("HEnergy", network)

    force = physics.GradientNode("forces", (henergy, positions), sign=-1)

    return henergy, force


def make_loss(henergy, force, force_training):
    """
    Build the loss graph for energy and force error.
    """
    from hippynn.graphs.nodes.loss import MSELoss, MAELoss, Rsq, Mean

    losses = {
        "T-RMSE": MSELoss.of_node(henergy) ** (1 / 2),
        "T-MAE": MAELoss.of_node(henergy),
        "T-RSQ": Rsq.of_node(henergy),
        "T-Hier": Mean.of_node(henergy.hierarchicality),
    }

    force_losses = {
        "F-RMSE": MSELoss.of_node(force) ** (1 / 2),
        "F-MAE": MAELoss.of_node(force),
        "F-RSQ": Rsq.of_node(force),
    }

    losses["EnergyTotal"] = losses["T-RMSE"] + losses["T-MAE"]
    losses["LossTotal"] = losses["EnergyTotal"] + losses["T-Hier"]
    if force_training:
        losses.update(force_losses)
        losses["ForceTotal"] = losses["F-RMSE"] + losses["F-MAE"]
        losses["LossTotal"] = losses["LossTotal"] + losses["ForceTotal"]

    return losses


# wb97x-6-31g*, G16. Doesn't need to be exact for most models, except atomization consistent.
# # # Old values with singlet/triplet multiplicity only
# # SELF_ENERGY_APPROX = {'C': -37.764142, 'H': -0.4993212, 'N': -54.4628753, 'O': -74.940046}
# Recalculated with appropriate vacuum multiplicity
SELF_ENERGY_APPROX = {"C": -37.8338334397, "H": -0.499321232710, "N": -54.5732824628, "O": -75.0424519384}
SELF_ENERGY_APPROX = {k: SELF_ENERGY_APPROX[v] for k, v in zip([6, 1, 7, 8], "CHNO")}
SELF_ENERGY_APPROX[0] = 0


def load_db(db_info, en_name, force_name, seed, anidata_location, n_workers, use_ccx_subset):
    """
    Load the database.
    """

    from hippynn.databases.h5_pyanitools import PyAniFileDB

    # Ensure total energies loaded in float64.
    torch.set_default_dtype(torch.float64)

    # Load DB, ensuring CCX energy info is available if that subset is selected.
    CCX_EN_NAME = "ccsd(t)_cbs.energy"
    if use_ccx_subset and en_name != CCX_EN_NAME:
        db_info["targets"].append(CCX_EN_NAME) # note, this is in-place and affects the evaluator.
    database = PyAniFileDB(file=anidata_location, species_key="atomic_numbers", seed=seed, num_workers=n_workers, **db_info)
    if use_ccx_subset and en_name != CCX_EN_NAME:
        database.targets.remove(CCX_EN_NAME) # undo in-place addition

    # compute (approximate) atomization energy by subtracting self energies

    # Build a lookup tensor for self energies
    max_z = max(SELF_ENERGY_APPROX.keys()) + 1  # +1 in case max Z is the last index
    lookup_table = torch.zeros(max_z, dtype=torch.float32)
    for z, energy in SELF_ENERGY_APPROX.items():
        lookup_table[z] = energy

    database.arr_dict["atomic_numbers"] = database.arr_dict["atomic_numbers"].long()

    self_energy = lookup_table[database.arr_dict["atomic_numbers"]]
    self_energy = self_energy.sum(dim=1)
    database.arr_dict[en_name] = database.arr_dict[en_name] - self_energy
    kcalpmol = ase.units.kcal / ase.units.mol
    conversion = ase.units.Ha / kcalpmol
    database.arr_dict[en_name] = database.arr_dict[en_name].float() * conversion

    if force_name in database.arr_dict:
        database.arr_dict[force_name] = database.arr_dict[force_name] * conversion
    torch.set_default_dtype(torch.float32)

    # Drop indices where computed energy not retrieved.
    if use_ccx_subset:
        filter_name = CCX_EN_NAME
    else:
        filter_name = en_name
    found_indices = ~torch.isnan(database.arr_dict[filter_name])
    database.arr_dict = {k: v[found_indices] for k, v in database.arr_dict.items()}
    database.make_trainvalidtest_split(test_size=0.1, valid_size=0.1)

    return database


def load_split(rank):
    from hippynn.experiment.serialization import restore_checkpoint

    with hippynn.active_directory("./data_ani1x_split", create=False):

        restarter_list = torch.load('restarters.pt', weights_only=False)
        restarter = restarter_list[rank]
        db = restarter.attempt_restart()
    return db


def setup_experiment(training_modules, device, batch_size, init_lr, patience, max_epochs, stopping_key):
    """
    Set up the training run.
    """
    from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController

    optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=init_lr)
    scheduler = RaiseBatchSizeOnPlateau(
        optimizer=optimizer,
        max_batch_size=batch_size,
        patience=patience,
        factor=0.5,
    )

    controller = PatienceController(
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        max_epochs=max_epochs,
        stopping_key=stopping_key,
        termination_patience=2 * patience,
    )

    setup_params = hippynn.experiment.SetupParams(
        controller=controller,
        device=device,
    )
    return setup_params


ANI1X_DSETS_KEYS = [
    "hf_tz.energy",
    "coordinates",
    "tpno_ccsd(t)_dz.corr_energy",
    "wb97x_dz.hirshfeld_charges",
    "wb97x_tz.mbis_charges",
    "wb97x_tz.forces",
    "mp2_tz.corr_energy",
    "npno_ccsd(t)_tz.corr_energy",
    "wb97x_tz.mbis_volumes",
    "wb97x_tz.energy",
    "wb97x_tz.dipole",
    "wb97x_tz.mbis_octupoles",
    "wb97x_tz.mbis_quadrupoles",
    "mp2_qz.corr_energy",
    "wb97x_tz.mbis_dipoles",
    "wb97x_dz.cm5_charges",
    "path",
    "atomic_numbers",
    "hf_qz.energy",
    "mp2_dz.corr_energy",
    "wb97x_dz.dipole",
    "npno_ccsd(t)_dz.corr_energy",
    "wb97x_dz.energy",
    "hf_dz.energy",
    "wb97x_dz.quadrupole",
    "ccsd(t)_cbs.energy",
    "wb97x_dz.forces",
]

AVAIL_METHODS = ["hf", "wb97x", "ccsd(t)", "mp2"]
AVAIL_BASIS = ["dz", "tz", "qz", "cbs"]


def get_data_names(qm_method, basis_set):
    assert qm_method in AVAIL_METHODS, f"Method not found: {qm_method}"
    assert basis_set in AVAIL_BASIS, f"Basis set not found: {basis_set}"
    data_spec = f"{qm_method}_{args.basis_set}"
    en_name = f"{data_spec}.energy"
    force_name = f"{data_spec}.forces"
    assert en_name in ANI1X_DSETS_KEYS, f"Method-basis combination not available: {data_spec}"
    if args.force_training:
        assert f"{data_spec}.forces" in ANI1X_DSETS_KEYS, f"Force training not available for data spec: {data_spec}"
    return en_name, force_name


def main(args):
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)

    hippynn.settings.WARN_LOW_DISTANCES = False
    if not args.progress:
        hippynn.settings.PROGRESS = None

    import os
    ntasks_per_node = int(os.environ.get('SLURM_NTASKS_PER_NODE', 1))
    n_nodes = int(os.environ.get('SLURM_NNODES', 1))
    this_rank = int(os.environ.get('SLURM_PROCID', 0))
    if args.scaling == "strong":
        num_devices_total = n_nodes * ntasks_per_node
        args.batch_size = args.batch_size//num_devices_total


    netname = f"{args.tag}_GPU{args.gpu}"
    network_parameters = {
        "possible_species": [0, 1, 6, 7, 8],
        "n_features": args.n_features,
        "n_sensitivities": args.n_sensitivities,
        "dist_soft_min": args.lower_cutoff,
        "dist_soft_max": args.cutoff_distance - 1,
        "dist_hard_max": args.cutoff_distance,
        "n_interaction_layers": args.n_interactions,
        "n_atom_layers": args.n_atom_layers,
    }

    with hippynn.tools.log_terminal("training_log.txt", "wt"):
        henergy, force = make_model(
            network_parameters,
            tensor_model=args.tensor_model,
            tensor_order=args.tensor_order,
            tensor_factors=args.tensor_factors,
            atomization_consistent=args.atomization_consistent,
        )

        en_name, force_name = get_data_names(args.qm_method, args.basis_set)

        henergy.mol_energy.db_name = en_name
        force.db_name = force_name

        validation_losses = make_loss(henergy, force, force_training=args.force_training)

        train_loss = validation_losses["LossTotal"]

        from hippynn.experiment import assemble_for_training

        training_modules, db_info = assemble_for_training(train_loss, validation_losses)

        database = load_split(rank=this_rank)            
        database.targets = db_info["targets"]
        database.inputs = db_info["inputs"]
        database.num_workers = args.n_workers

        from hippynn.pretraining import hierarchical_energy_initialization

        hierarchical_energy_initialization(henergy, database, trainable_after=False)

        patience = args.patience
        if args.use_ccx_subset:
            patience *= 4

        setup_params = setup_experiment(
            training_modules,
            device=args.gpu,
            batch_size=args.batch_size,
            init_lr=args.init_lr,
            patience=patience,
            max_epochs=args.max_epochs,
            stopping_key=args.stopping_key,
        )

    from hippynn.experiment import HippynnLightningModule
    lightmod, datamodule = HippynnLightningModule.from_experiment_setup(training_modules, database, setup_params)

    from lightning.pytorch.loggers import CSVLogger
    logger = CSVLogger(save_dir=".", name=netname)
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpointer = ModelCheckpoint(monitor=f"valid_{args.stopping_key}",
                                   save_last=True,
                                   save_top_k=5,
                                   every_n_epochs=50,
                                   every_n_train_steps=None,
                                   )

    import pytorch_lightning as pl

    accelerator='cpu'
    if args.use_gpu:
        if torch.cuda.is_available():
            accelerator='gpu'
        else:
            print("Cuda not available, using CPU")
            

    trainer = pl.Trainer(accelerator=accelerator,
                         devices=ntasks_per_node,
                         logger=logger,
                         num_nodes=n_nodes,
                         log_every_n_steps=100,
                         callbacks=[checkpointer],
                         max_epochs=1000000000, # hippynn terminates training
                         ) #'auto' detects MPS which doesn't work.
    #lightmod.model.print_structure()
    trainer.fit(model=lightmod, datamodule=datamodule)
    return
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    from argparse import BooleanOptionalAction

    parser.add_argument("--tag", type=str, default="TEST_MODEL_ANI1X", help="name for run")
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to run on, if any")
    parser.add_argument(
        "--use-gpu",
        action=BooleanOptionalAction,
        default=torch.cuda.is_available(),
        help="Whether to use GPU. Defaults to torch.cuda.is_available()",
    )

    parser.add_argument("--seed", type=int, default=0, help="random seed for init and split")

    parser.add_argument("--n_interactions", type=int, default=2)
    parser.add_argument("--n_atom_layers", type=int, default=3)
    parser.add_argument("--n_features", type=int, default=128)
    parser.add_argument("--n_sensitivities", type=int, default=20)
    parser.add_argument("--cutoff_distance", type=float, default=6.5)
    parser.add_argument("--lower_cutoff", type=float, default=0.75, help="Where to initialize the shortest distance sensitivity")
    parser.add_argument(
        "--tensor_model",
        type=str.upper,
        default="HOP",
        choices=["HOP", "TS", "NONE"],
        help="Which tensor architecture to use.  Choices are 'HOP' for HIP-HOP-NN, "
        "'TS' for HIP-NN-TS, and 'NONE' for vanilla HIP-NN'. "
        "If tensor_order==0 then vanilla HIP-NN will "
        "be used regardless.",
    )
    parser.add_argument("--tensor_order", type=int, default=0, help="tensor order $\\ell$")
    parser.add_argument("--tensor_factors", type=int, default=4, help="number of factors used (in HIP-HOP-NN only)")
    parser.add_argument("--atomization_consistent", type=bool, default=False)

    parser.add_argument("--anidata_location", type=str, default="../../../datasets/ani1x_release/ani1x-release.h5")
    parser.add_argument("--qm_method", type=str, default="wb97x")
    parser.add_argument("--basis_set", type=str, default="dz")
    parser.add_argument("--force_training", action=BooleanOptionalAction, default=True, help="Use force training.")
    parser.add_argument("--scaling",type=str, choices=['strong','weak'], default='strong',help="strong or weak scaling of batch size")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--stopping_key", type=str, default="T-RMSE")

    parser.add_argument(
        "--use_ccx_subset",
        type=bool,
        default=False,
        help="Train only to configurations from the ANI-1ccx subset."
        " Note that this will still use the energies using the `qm_method` argument."
        " *Note!* This argument will multiply the patience by a factor of 4.",
    )

    parser.add_argument("--progress", action=BooleanOptionalAction, default=True, help="Whether to use progress bars.")
    parser.add_argument("--n_workers", type=int, default=2, help="workers for pytorch dataloaders")
    args = parser.parse_args()

    main(args)
