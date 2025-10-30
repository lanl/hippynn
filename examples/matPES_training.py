"""
This script uses the MatPES dataset introduced in:
"A Foundational Potential Energy Surface Dataset for Materials"
(arXiv:2503.04070, https://arxiv.org/abs/2503.04070).

For more details and raw files see https://matpes.ai/ .

To convert the raw files into a hippynn-compatible format,
use the example file convert_matPES.py .
"""


import argparse
import torch
import hippynn


def make_model_and_loss(network_params, tensor_model, tensor_order, tensor_factors, atomization_consistent, force_training):
    """
    Build the model graph for energy and potentially force prediction.
    """
    from hippynn.graphs import inputs, networks, targets, physics
    from hippynn.graphs.nodes.misc import StrainInducer

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

    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")
    cell = inputs.CellNode(db_name="C")

    network = net_class("nn", (species, positions, cell), module_kwargs=network_params, periodic=True)

    if not atomization_consistent:
        energy = targets.HEnergyNode("HEnergy", network)
    else:
        energy = targets.AtomizationEnergyNode("HEnergy", network)

    if args.force_training:

        stress, forces = physics.setup_stressforce_nodes(energy)

        forces.db_name = "F"

        import ase.units

        # Stress is natively in eV/atom. We need to apply the conversion factor to kbar.
        # VASP sign convention for stress is opposite of ASE (which we use).
        kbar = ase.units.GPa / 10
        stress = (-1/kbar) * stress
        stress.db_name = "stress"

    energy.mol_energy.db_name = "T"
    energy_peratom = physics.PerAtom("energy_peratom", energy, db_name="t_peratom")

    """
    Build the loss graph for energy and force error.
    """
    from hippynn.graphs.nodes.loss import MSELoss, MAELoss, Rsq, Mean

    losses = {
        "T-Hier": Mean.of_node(energy.hierarchicality),
    }
    en_nodes = {
        "T": energy,
        "t": energy_peratom,
    }
    for ename, enode in en_nodes.items():
        losses[ename + "-MAE"] = MAELoss.of_node(enode)

    if args.force_training:
        losses["F-MAE"] = MAELoss.of_node(forces)
        losses["s-MAE"] = MAELoss.of_node(stress)

    losses["EnergyTotal"] = losses["T-MAE"]
    if args.force_training:
        losses["ForceTotal"] = losses["F-MAE"]
        losses["StressTotal"] = 0.1 * losses["s-MAE"]

    losses["LossTotal"] = losses["EnergyTotal"] + 0.01 * losses["T-Hier"]
    if force_training:
        losses["LossTotal"] = losses["LossTotal"] + losses["ForceTotal"] + losses["StressTotal"]

    return energy, losses


def load_db(db_info, delete_fraction, seed, data_location, n_workers):
    """
    Load the database.
    """

    from hippynn.databases import NPZDatabase

    database = NPZDatabase(file=data_location, seed=seed, num_workers=n_workers, allow_unfound=True, pin_memory=False, **db_info)

    if delete_fraction:
        n_items = len(database.arr_dict["E"])
        remove = int(n_items * delete_fraction)
        for k, v in database.arr_dict.items():
            database.arr_dict[k] = v[:-remove]

    for sname in ["train", "valid", "test"]:
        mask = database.arr_dict[sname]
        database.make_explicit_split_bool(sname, mask)

    return database


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


def validate_species(database, possible_species):
    split_species = [torch.unique(split['Z']) for name, split in database.splits.items()]
    all_species = torch.unique(torch.cat(split_species))
    possible_species = torch.as_tensor(possible_species)
    try:
        assert torch.isin(all_species, possible_species).all(), "Possible species was wrong!"
    except AssertionError as ae:
        print("Actual species values:", all_species)
        print("Expected species values:", possible_species)
        raise ae


def main(args):
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.set_device(args.gpu)
    else:
        torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)

    hippynn.settings.WARN_LOW_DISTANCES = False
    if not args.progress:
        hippynn.settings.PROGRESS = None

    netname = f"{args.tag}_GPU{args.gpu}"
    network_parameters = {
        "possible_species": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                             11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                             21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                             31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                             41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                             51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                             61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                             71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                             81, 82, 83, 89, 90, 91, 92, 93, 94, ],  # atoms in R2SCAN version of matPES
        "n_features": args.n_features,
        "n_sensitivities": args.n_sensitivities,
        "dist_soft_min": args.lower_cutoff,
        "dist_soft_max": args.cutoff_distance - 1,
        "dist_hard_max": args.cutoff_distance,
        "n_interaction_layers": args.n_interactions,
        "n_atom_layers": args.n_atom_layers,
    }

    with hippynn.tools.active_directory(netname):
        with hippynn.tools.log_terminal("training_log.txt", "wt"):
            energy, validation_losses = make_model_and_loss(
                network_parameters,
                tensor_model=args.tensor_model,
                tensor_order=args.tensor_order,
                tensor_factors=args.tensor_factors,
                atomization_consistent=args.atomization_consistent,
                force_training=args.force_training,
            )

            train_loss = validation_losses["LossTotal"]

            from hippynn.experiment import assemble_for_training

            training_modules, db_info = assemble_for_training(train_loss, validation_losses)

            database = load_db(
                db_info,
                delete_fraction=args.delete_fraction,
                n_workers=args.n_workers,
                seed=args.seed,
                data_location=args.data_location,
            )

            validate_species(database,network_parameters["possible_species"])

            from hippynn.pretraining import hierarchical_energy_initialization

            hierarchical_energy_initialization(energy, database, trainable_after=False)

            if args.precompute_pairs:
                from hippynn.experiment.assembly import precompute_pairs

                print("Precomputing pairs.")

                precompute_pairs(training_modules.model, database, n_images=3, batch_size=4)
                training_modules, db_info = assemble_for_training(train_loss, validation_losses)  # , plot_maker=plot_maker)
                database.inputs = db_info["inputs"]
                if args.use_gpu:
                    database.send_to_device()
            else:
                print("Not precomputing pairs.")

            patience = args.patience

            setup_params = setup_experiment(
                training_modules,
                device=args.gpu if args.use_gpu else "cpu",
                batch_size=args.batch_size,
                init_lr=args.init_lr,
                patience=patience,
                max_epochs=args.max_epochs,
                stopping_key=args.stopping_key,
            )

            from hippynn.experiment import setup_and_train

            setup_and_train(
                training_modules=training_modules,
                database=database,
                setup_params=setup_params,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    from argparse import BooleanOptionalAction

    torch.cuda.is_available()

    parser.add_argument("--tag", type=str, default="TEST_MODEL_MatPES", help="name for run")
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to run on")
    parser.add_argument(
        "--use-gpu",
        action=BooleanOptionalAction,
        default=torch.cuda.is_available(),
        help="Whether to use GPU. Defaults to torch.cuda.is_available()",
    )

    parser.add_argument("--seed", type=int, default=0, help="random seed for network initialization")

    parser.add_argument("--n_interactions", type=int, default=2)
    parser.add_argument("--n_atom_layers", type=int, default=5)
    parser.add_argument("--n_features", type=int, default=128)
    parser.add_argument("--n_sensitivities", type=int, default=20)
    parser.add_argument("--cutoff_distance", type=float, default=6.0)
    parser.add_argument("--lower_cutoff", type=float, default=0.8, help="Where to initialize the shortest distance sensitivity")
    parser.add_argument(
        "--tensor_model",
        type=str.upper,
        default="HOP",
        choices=["HOP", "TS", "NONE"],
        help="Which tensor architecture to use.  Choices are 'HOP' for HIP-HOP-NN, " "'TS' for HIP-NN-TS, and 'NONE' for vanilla HIP-NN'. "
             "If tensor_order==0 then vanilla HIP-NN will be used regardless.",
    )
    parser.add_argument("--tensor_order", type=int, default=2, help="tensor order $\\ell$")
    parser.add_argument("--tensor_factors", type=int, default=3, help="number of factors used (in HIP-HOP-NN only)")
    parser.add_argument("--atomization_consistent", type=bool, default=False)

    parser.add_argument("--data_location", type=str, default="../../../datasets/matPES/matPES_R2SCAN.npz")
    parser.add_argument("--delete_fraction", type=float, default=0.0,
                        help="For debugging. Removes this fraction of the database before training.")

    parser.add_argument("--force_training", action=BooleanOptionalAction, default=True, help="Use force/stress training.")

    parser.add_argument(
        "--precompute_pairs",
        action=BooleanOptionalAction,
        default=False,
        help="Whether to precompute and store pairs once before training. Default false.",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--stopping_key", type=str, default="t-MAE")

    parser.add_argument("--progress", action=BooleanOptionalAction, default=True, help="Whether to use progress bars.")
    parser.add_argument("--n_workers", type=int, default=0, help="workers for pytorch dataloaders")
    args = parser.parse_args()

    print("Using script arguments:")
    for k, v in vars(args).items():
        print(k, ":", v)
    main(args)
