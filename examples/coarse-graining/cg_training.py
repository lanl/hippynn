import os
import itertools

import numpy as np
import torch

from hippynn.molecular_dynamics.misc import SpeciesLookup
from hippynn.databases import NPZDatabase
from hippynn.experiment import SetupParams, setup_and_train
from hippynn.experiment.assembly import assemble_for_training
from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController
from hippynn.graphs import IdxType
from hippynn.graphs.nodes import loss
from hippynn.graphs.nodes.indexers import acquire_encoding_padding, SpeciesIndexer
from hippynn.graphs.nodes.inputs import SpeciesNode, PositionsNode, CellNode
from hippynn.graphs.nodes.networks import HipnnQuad
from hippynn.graphs.nodes.pairs import KDTreePairsMemory
from hippynn.graphs.nodes.physics import MultiGradientNode
from hippynn.graphs.nodes.targets import HEnergyNode
from hippynn.plotting import PlotMaker, Hist2D, SensitivityPlot
from hippynn.tools import active_directory, log_terminal

from repulsive_potential import RepulsivePotentialBySpeciesNode
from save_model_for_lammps import save_model_for_lammps

training_data_file = os.path.join(os.pardir,os.pardir,os.pardir,"datasets","cg_methanol_trajectory.npz")
training_data_file = os.path.abspath(training_data_file)

with np.load(training_data_file, allow_pickle=True) as data:
    # For the repulsive potential, we need to provide a Tensor where entry [i, j] corresponds 
    # to the taper point for the repulsive potential between particles of type i and j. If we 
    # don't have species specific RDF information, we'll just set this value to be the same 
    # for all species pairs.
    species = data["species"]
    rdf_bins = data["rdf_bins"]

    unique_species = np.unique(species)
    unique_species = sorted(list(unique_species))

    repulsive_potential_taper_point = torch.zeros((max(unique_species)+1, max(unique_species)+1))

    def taper_point_fn(rdf_bins, rdf_values):
        return rdf_bins[np.where(rdf_values > 0.001)[0][0]]

    try:
        for type1, type2 in itertools.combinations_with_replacement(unique_species, 2):
            rdf_values = data[f"rdf_values_{type1}_{type2}"]
            taper_point = taper_point_fn(rdf_bins, rdf_values)
            repulsive_potential_taper_point[type1, type2] = taper_point
            repulsive_potential_taper_point[type2, type1] = taper_point
    except KeyError:
        # if species RDF data is missing, we'll use the full RDF and the same value for each species pair
        rdf_values = data["rdf_values"]
        repulsive_potential_taper_point = taper_point_fn(rdf_bins, rdf_values) + torch.zeros((max(unique_species)+1, max(unique_species)+1))

    repulsive_potential_strength = np.abs(data["forces"]).mean()

    try:
        species_lookup = SpeciesLookup(data["species"], data["species_names"])
    except KeyError:
        species_lookup = None

    # the data for 'species' must be of shape (n_frames, n_atoms)
    n_frames, n_atoms, _ = data["positions"].shape
    if data["species"].squeeze().shape == (n_atoms,):
        data["species"] = np.tile(data["species"], (n_frames, 1))
        np.savez()

results_folder = "model"

with active_directory(results_folder):
    with log_terminal("training_log.txt", "wt"):

        ## Initialize needed nodes for network
        # Network input nodes
        species = SpeciesNode(name="species", db_name="species")
        positions = PositionsNode(name="positions", db_name="positions")
        cells = CellNode(name="cells", db_name="cells")

        # Network hyperparameters
        network_params = {
            "possible_species": [0] + unique_species, # hippynn requires a sentinal/null species of 0
            "n_features": 128,
            "n_sensitivities": 20,
            "dist_soft_min": 2.0,
            "dist_soft_max": 13.0,
            "dist_hard_max": 15.0,
            "n_interaction_layers": 1,
            "n_atom_layers": 3,
            "sensitivity_type": "inverse",
            "resnet": True,
        }

        # Species encoder
        enc, pdx = acquire_encoding_padding([species], species_set=[0] + unique_species)

        # Pair finder
        pair_finder = KDTreePairsMemory(
            "pairs",
            (positions, enc, pdx, cells),
            dist_hard_max=network_params["dist_hard_max"],
            skin=0,
        )

        # HIP-NN-TS node with l=2
        network = HipnnQuad(
            "HIPNN", (pdx, pair_finder), module_kwargs=network_params, periodic=True
        )

        # Network energy prediction
        henergy = HEnergyNode("HEnergy", parents=(network,))

        # Repulsive potential
        repulse = RepulsivePotentialBySpeciesNode(
            "repulse", 
            (pair_finder, pdx), 
            taper_point=repulsive_potential_taper_point,
            strength=repulsive_potential_strength,
            dr=0.15,
            perc=0.05,
        )

        # Combined energy prediction
        sys_energy = henergy.mol_energy + repulse.mol_energy
        sys_energy.name = "sys_energy"
        sys_energy.index_state = IdxType.Systems

        # Force node
        grad = MultiGradientNode("forces", sys_energy, (positions,), signs=-1)
        force = grad.children[0]
        force.db_name = "forces"

        # Now that we've constructed our model, we'll build the loss metrics
        validation_losses = {}

        # This will allow us to calculate losses and make plots of the losses by species
        # split force by species
        force_pred_by_species = SpeciesIndexer("force_by_species_pred", parents=(force.pred,))
        force_true_by_species = SpeciesIndexer("force_by_species_true", parents=(force.true,))

        for true_idxed, pred_idxed in zip(force_true_by_species.children, force_pred_by_species.children):
            species_id = true_idxed.name.split("_")[-1] # Parse the node name to find the species value
            if species_lookup is not None:
                species_id = species_lookup.number_to_name(species_id) # get name corresponding to species number
            validation_losses.update(
                {
                    f"ForceRMSESpecies{species_id}": loss.MSELoss(pred_idxed, true_idxed) ** (1 / 2),
                    f"ForceMAESpecies{species_id}": loss.MAELoss(pred_idxed, true_idxed),
                    f"ForceRsqSpecies{species_id}": loss.Rsq(pred_idxed, true_idxed),
                }
            )

        # System-wide losses
        force_rsq = loss.Rsq.of_node(force)
        force_rmse = loss.MSELoss.of_node(force) ** (1 / 2)
        force_mae = loss.MAELoss.of_node(force)
        total_loss = force_rmse + force_mae

        validation_losses.update({
            "ForceRMSE": force_rmse,
            "ForceMAE": force_mae,
            "ForceRsq": force_rsq,
            "TotalLoss": total_loss,
        })

        plotters = [
            Hist2D.compare(force, saved="forces", shown=False),
            SensitivityPlot(
                network.torch_module.sensitivity_layers[0], saved="sensitivity", shown=False
            ),
        ]

        for true_idxed, pred_idxed in zip(force_true_by_species.children, force_pred_by_species.children):
            species_id = true_idxed.name.split("_")[-1] # Parse the node name to find the species value
            if species_lookup is not None:
                species_id = species_lookup.number_to_name(species_id) # get name corresponding to species number
            plotters.append(
                Hist2D(
                    x_var=true_idxed, 
                    y_var=pred_idxed, 
                    xlabel=f"true force, species {species_id}", 
                    ylabel=f"predicted force, species {species_id}", 
                    saved=f"force_species_{species_id.replace(' ', '_')}",
                )
            )

        plot_maker = PlotMaker(
            *plotters,
            plot_every=10,
        )

        ## Build network
        training_modules, db_info = assemble_for_training(
            total_loss, validation_losses, plot_maker=plot_maker
        )

        ## Load training data
        database = NPZDatabase(
            training_data_file, 
            seed=0, 
            **db_info, 
            valid_size=0.1, 
            test_size=0.1,
        )

        ## Set up optimizer
        optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=1e-3)

        scheduler = RaiseBatchSizeOnPlateau(
            optimizer=optimizer,
            max_batch_size=64,
            patience=10,
            factor=0.5,
        )

        controller = PatienceController(
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=1,
            fraction_train_eval=0.2,
            eval_batch_size=1,
            max_epochs=200,
            termination_patience=20,
            stopping_key="TotalLoss",
        )

        experiment_params = SetupParams(controller=controller)

        ## Train!
        metric_tracker = setup_and_train(
            training_modules=training_modules,
            database=database,
            setup_params=experiment_params,
        )

        print(f"PyTorch model saved in directory {os.path.abspath(results_folder)}")

        # To save a version to run in LAMMPS, the species names must be ordered corresponding to the 
        # list `possible_species` provided as a network parameter (without the 0)
        if species_lookup is not None:
            species_names_ordered = [species_lookup.number_to_name(num) for num in unique_species]
        else:
            species_names_ordered = ["MeOH"] # for methanol example

        try:
            save_model_for_lammps(model_folder=".", species_names_ordered=species_names_ordered)
        except ImportError as e:
            print(f"Unable to save model as LAMMPS ML-IAP model: {e}.")