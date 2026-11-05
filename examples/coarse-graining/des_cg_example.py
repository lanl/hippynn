"""
Coarse-grained (CG) force field for deep eutectic solvents (DES).

This script trains a HIP-HOP-NN CG model to the CG-mapped forces of an all-atom (AA)
DES trajectory, runs CG MD with the hippynn MD driver, and compares the CG partial
radial distribution functions (RDFs) to the AA reference.

BEFORE RUNNING:
Download the DES dataset and place its .npz files in a folder called datasets/des_cg/ at the
same level as hippynn/, or change ``data_dir`` below. Choose ``system``, ``temperature``, and
``mapping`` below. The Martini mapping (reline and hydrated reline only) splits choline into two
beads, Ch1 and Ch2, and the model is trained on all four bead types. After the CG MD, the two
choline beads are merged back into one at their center of mass so that the RDFs can be compared
to the one-bead-per-molecule reference and paper CG data. Hydrated reline (250 choline, 250
chloride, 500 urea, 84 water; water removed before coarse-graining) is only available at 298 K
with the COM and Martini mappings.

The repulsive potential settings and the number of epochs follow the paper models. Ethaline, reline
298 K COM and reline 333 K COG used degree 2 and strength mean(|F_x|)/10; the other
reline models, including Martini and hydrated reline, used degree 3 and strength mean(|F|)/10.
Results may differ slightly from the stored RDFs, since the seeds differ.

Approx. GPU performance (one A100): training 30-60 min; MD ~5 h for the paper-length run
(0.1 ns burn-in + 1 ns production). Lower ``n_burnin_steps`` and ``n_production_steps``
for a quick test. The analysis computes the RDF over every recorded frame, which takes
~30 min for 10,000 frames on one CPU core; set ``rdf_stride`` to use fewer frames.
"""

import os
import itertools

import numpy as np
import torch
import matplotlib.pyplot as plt
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from scipy.signal import find_peaks

from hippynn.databases import Database
from hippynn.experiment import SetupParams, setup_and_train
from hippynn.experiment.assembly import assemble_for_training
from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController
from hippynn.experiment.serialization import load_checkpoint_from_cwd
from hippynn.graphs import IdxType
from hippynn.graphs.nodes import loss
from hippynn.graphs.nodes.indexers import acquire_encoding_padding, SpeciesIndexer
from hippynn.graphs.nodes.inputs import SpeciesNode, PositionsNode, CellNode
from hippynn.graphs.nodes.networks import HipHopnn
from hippynn.graphs.nodes.pairs import KDTreePairsMemory
from hippynn.graphs.nodes.physics import MultiGradientNode
from hippynn.graphs.nodes.targets import HEnergyNode
from hippynn.graphs.predictor import Predictor
from hippynn.molecular_dynamics.md import Variable, NullUpdater, ASELangevinDynamics, MolecularDynamics
from hippynn.molecular_dynamics.rdf import calculate_rdf
from hippynn.plotting import PlotMaker, Hist2D, SensitivityPlot
from hippynn.tools import active_directory, log_terminal

from repulsive_potential import RepulsivePotentialPolynomialBySpeciesNode, find_taper_point

AA_COLOR, CG_PAPER_COLOR, CG_RUN_COLOR = "#0072B2", "#E69F00", "#D55E00"

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- User parameters -----
system = "ethaline"  # "reline", "reline_hydrated", or "ethaline"
temperature = 298  # 298 or 333 (K)
mapping = "COM"  # "COG", "COM", or "Martini" (reline and reline_hydrated only)
data_dir = os.path.abspath(os.path.join(os.pardir, os.pardir, os.pardir, "datasets", "des_cg"))

run_training = True
run_md = True
run_analysis = True  # computes the CG RDFs and saves cg_rdfs.npz
run_plotting = True  # compares RDFs and plots; needs the saved cg_rdfs.npz

n_burnin_steps = 100_000
n_production_steps = 1_000_000
record_every = 100
rdf_stride = 1  # use every n-th recorded frame in the RDF analysis
# ---------------------------

assert system in ("reline", "reline_hydrated", "ethaline"), "Only 'reline', 'reline_hydrated', and 'ethaline' are possible systems."
assert mapping in ("COG", "COM", "Martini"), "Only the COG, COM, and Martini mappings are available."
assert not (mapping == "Martini" and system == "ethaline"), "The Martini mapping is only available for reline."
assert not (system == "reline_hydrated" and (temperature != 298 or mapping == "COG")), "Hydrated reline is only available at 298 K with COM or Martini."

# Repulsive potential settings and epochs of the paper models
paper_group_a = system == "ethaline" or (system, temperature, mapping) in [("reline", 298, "COM"), ("reline", 333, "COG")]
rp_degree = 2 if paper_group_a else 3
max_epochs = 80 if paper_group_a else 20

# Species of the (merged) one-bead-per-molecule systems, used to compare RDFs. The Martini model has four bead types.
analysis_species_names = {1: "Cho", 2: "Cl", 3: ("EG" if system == "ethaline" else "Urea")}
species_names = {1: "Ch1", 2: "Ch2", 3: "Cl", 4: "Urea"} if mapping == "Martini" else analysis_species_names
unique_species = sorted(species_names)
results_folder = f"{system}_{temperature}_{mapping}"
aa_file = os.path.join(data_dir, f"{system}_{temperature}_{mapping}_aa.npz")
paper_cg_file = os.path.join(data_dir, f"{system}_{temperature}_{mapping}_cg.npz")

ps = 1000 * units.fs  # ase.units has no picosecond
temperature_K = (temperature + 0.15 if temperature in [298, 333] else temperature)

## Load the AA reference (CG-mapped) data
with np.load(aa_file) as data:
    # Every 10th frame (10 ps spacing) for training
    frames = slice(0, None, 10)
    arr_dict = {
        "species": data["species"][frames],
        "positions": data["position"][frames],
        "cells": data["cells"][frames],
        "forces": data["force"][frames],
    }
    masses = data["masses"][frames]
    rdf_bins = data["rdf_bins"]
    aa_rdfs = {key: data[key] for key in data.files if key.startswith("rdf_values_")}
    # Reference RDFs for the comparison; for Martini these are between the merged choline, chloride, and urea beads
    ref_prefix = "merged_rdf_values_" if mapping == "Martini" else "rdf_values_"
    ref_rdfs = {key.replace("merged_", ""): data[key] for key in data.files if key.startswith(ref_prefix)}

if run_training:
    with active_directory(results_folder):
        with log_terminal("training_log.txt", "wt"):

            ## Repulsive potential parameters from the AA RDFs
            # The repulsion acts on the rising edge of the first RDF peak: zero force at its midpoint
            # (taper point) and full strength at the onset, where g(r) first exceeds 0.01.
            def rdf_onset_and_peak(rdf_bins, rdf_values):
                peaks, _ = find_peaks(rdf_values, prominence=0.05)
                return rdf_bins[np.where(rdf_values > 0.01)[0][0]], rdf_bins[peaks[0]]

            def taper_point_fn(rdf_bins, rdf_values):
                onset, peak = rdf_onset_and_peak(rdf_bins, rdf_values)
                return float(peak + onset) / 2

            def dr_fn(rdf_bins, rdf_values):
                onset, peak = rdf_onset_and_peak(rdf_bins, rdf_values)
                return float(peak - onset) / 2

            repulsive_potential_taper_point = find_taper_point(unique_species, aa_rdfs, rdf_bins, taper_point_fn)
            repulsive_potential_dr = find_taper_point(unique_species, aa_rdfs, rdf_bins, dr_fn)
            # Placeholder width for the unused null species, avoids division by zero
            repulsive_potential_dr[repulsive_potential_dr == 0] = 1.0
            if paper_group_a:
                repulsive_potential_strength = np.abs(arr_dict["forces"]).mean() / 10
            else:
                repulsive_potential_strength = np.linalg.norm(arr_dict["forces"], axis=-1).mean() / 10

            print(f"Repulsive potential taper points:\n{repulsive_potential_taper_point}")
            print(f"Repulsive potential widths:\n{repulsive_potential_dr}")
            print(f"Repulsive potential strength: {repulsive_potential_strength}")

            ## Initialize needed nodes for network
            # Network input nodes
            species = SpeciesNode(name="species", db_name="species")
            positions = PositionsNode(name="positions", db_name="positions")
            cells = CellNode(name="cells", db_name="cells")

            # Network hyperparameters
            network_params = {
                "possible_species": [0] + unique_species,  # hippynn requires a sentinal/null species of 0
                "n_features": 32,
                "n_sensitivities": 20,
                "dist_soft_min": 2.5,
                "dist_soft_max": 13.0,
                "dist_hard_max": 15.0,
                "n_interaction_layers": 1,
                "n_atom_layers": 3,
                "sensitivity_type": "inverse",
                "resnet": True,
                "l_max": 2,
                "n_max": 3,
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

            # HIP-HOP-NN node
            network = HipHopnn("HIPNN", (pdx, pair_finder), module_kwargs=network_params, periodic=True)

            # Network energy prediction
            henergy = HEnergyNode("HEnergy", parents=(network,))

            # Repulsive potential
            repulse = RepulsivePotentialPolynomialBySpeciesNode(
                "repulse",
                (pair_finder, pdx),
                taper_point=repulsive_potential_taper_point,
                strength=repulsive_potential_strength,
                dr=repulsive_potential_dr,
                degree=rp_degree,
            )

            # Combined energy prediction
            sys_energy = henergy.mol_energy + repulse.mol_energy
            sys_energy.name = "sys_energy"
            sys_energy.index_state = IdxType.Systems

            # Force node
            grad = MultiGradientNode("forces", sys_energy, (positions,), signs=-1)
            force = grad.children[0]
            force.db_name = "forces"

            ## Losses
            validation_losses = {}

            # split force by species
            force_pred_by_species = SpeciesIndexer("force_by_species_pred", parents=(force.pred,))
            force_true_by_species = SpeciesIndexer("force_by_species_true", parents=(force.true,))

            plotters = [
                Hist2D.compare(force, saved="forces", shown=False),
                SensitivityPlot(network.torch_module.sensitivity_layers[0], saved="sensitivity", shown=False),
            ]

            for true_idxed, pred_idxed in zip(force_true_by_species.children, force_pred_by_species.children):
                species_name = species_names[int(true_idxed.name.split("_")[-1])]  # Parse the node name to find the species value
                validation_losses.update(
                    {
                        f"ForceRMSE{species_name}": loss.MSELoss(pred_idxed, true_idxed) ** (1 / 2),
                        f"ForceMAE{species_name}": loss.MAELoss(pred_idxed, true_idxed),
                        f"ForceRsq{species_name}": loss.Rsq(pred_idxed, true_idxed),
                    }
                )
                plotters.append(
                    Hist2D(
                        x_var=true_idxed,
                        y_var=pred_idxed,
                        xlabel=f"true force, {species_name}",
                        ylabel=f"predicted force, {species_name}",
                        saved=f"force_{species_name}",
                    )
                )

            # System-wide losses
            force_rsq = loss.Rsq.of_node(force)
            force_rmse = loss.MSELoss.of_node(force) ** (1 / 2)
            force_mae = loss.MAELoss.of_node(force)

            validation_losses.update(
                {
                    "ForceRMSE": force_rmse,
                    "ForceMAE": force_mae,
                    "ForceRsq": force_rsq,
                }
            )

            plot_maker = PlotMaker(*plotters, plot_every=10)

            ## Build network
            training_modules, db_info = assemble_for_training(force_mae, validation_losses, plot_maker=plot_maker)

            ## Load training data
            database = Database(arr_dict, seed=0, **db_info, valid_size=0.1, test_size=0.1)

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
                max_epochs=max_epochs,
                termination_patience=20,
                stopping_key="ForceMAE",
            )

            experiment_params = SetupParams(controller=controller)

            ## Train!
            metric_tracker = setup_and_train(
                training_modules=training_modules,
                database=database,
                setup_params=experiment_params,
            )

            print(f"PyTorch model saved in directory {os.path.abspath(os.curdir)}")

if run_md:
    with active_directory(results_folder):
        with log_terminal("md_log.txt", "wt"):

            ## Initial conditions: a random training frame with Maxwell-Boltzmann velocities at the target temperature
            rng = np.random.default_rng(0)
            start = rng.integers(len(arr_dict["positions"]))
            print(f"Starting from training frame {start}")

            atoms = Atoms(
                positions=arr_dict["positions"][start],
                masses=masses[start].squeeze(-1),
                cell=arr_dict["cells"][start],
                pbc=True,
            )
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(0))
            velocity = atoms.get_velocities() / (units.Ang / ps)  # ASE velocity units to Å/ps

            def to_tensor(array, dtype=torch.float32):
                return torch.as_tensor(array, dtype=dtype, device=device)[None, ...]

            cell = to_tensor(arr_dict["cells"][start])
            velocity = to_tensor(velocity)

            positions_variable = Variable(
                name="positions",
                data={
                    "position": to_tensor(arr_dict["positions"][start]),
                    "velocity": velocity,
                    "mass": to_tensor(masses[start]),
                    "acceleration": torch.zeros_like(velocity),
                    "cell": cell,
                },
                model_input_map={"positions": "position"},
                device=device,
            )

            positions_variable.updater = ASELangevinDynamics(
                force_db_name="forces",
                temperature_K=temperature_K,
                frix=20,  # in 1/ps since the time unit below is ps
                force_units=units.kcal / units.mol / units.Ang,  # must match the training data
                position_units=units.Ang,  # must match the training data
                time_units=ps,  # the velocity data is in Å/ps
                seed=0,
            )

            cell_variable = Variable(
                name="cell",
                data={"cell": cell},
                model_input_map={"cells": "cell"},
                device=device,
                updater=NullUpdater(),
            )

            species_variable = Variable(
                name="species",
                data={"species": to_tensor(arr_dict["species"][start], dtype=torch.int)},
                model_input_map={"species": "species"},
                device=device,
                updater=NullUpdater(),
            )

            ## Load model
            check = load_checkpoint_from_cwd(model_device=device, restart_db=False)
            model = Predictor.from_graph(check["training_modules"].model)
            model.to(torch.float32)
            model.to(device)

            pairs = model.graph.unique_node_from_name("pairs")
            pairs.skin = 3  # see hippynn.graphs.nodes.pairs.KDTreePairsMemory documentation

            ## Run MD
            emdee = MolecularDynamics(
                variables=[positions_variable, species_variable, cell_variable],
                model=model,
            )

            # time units are those given to the VariableUpdater, i.e. ps
            emdee.run(dt=0.001, n_steps=n_burnin_steps)
            emdee.run(dt=0.001, n_steps=n_production_steps, record_every=record_every)

            ## Save results with the same keys as the dataset files
            data = {key: value.cpu().numpy().squeeze(1) for key, value in emdee.get_data().items()}
            np.savez(
                "cg_trajectory.npz",
                position=data["positions_position"],
                velocity=data["positions_velocity"],
                force=data["positions_force"],
                masses=data["positions_mass"],
                cells=data["positions_cell"],
                species=data["species_species"],
            )
            print(f"CG trajectory saved at {os.path.abspath('cg_trajectory.npz')}")

def merge_choline_beads(positions, masses, cells, species):
    """Merge each Ch1 (species 1) and Ch2 (species 2) bead into one choline bead at their center of mass.

    The k-th Ch1 bead is paired with the k-th Ch2 bead. Returns positions and species ordered as choline (1),
    chloride (2), urea (3).
    """
    masses = masses.reshape(len(masses), -1)[0]
    ch1, ch2, cl, urea = (species[0] == s for s in (1, 2, 3, 4))
    box = np.diagonal(cells, axis1=1, axis2=2)[:, None, :]
    separation = positions[:, ch2] - positions[:, ch1]
    separation -= box * np.round(separation / box)  # minimum image
    m1, m2 = masses[ch1][:, None], masses[ch2][:, None]
    choline = positions[:, ch1] + separation * m2 / (m1 + m2)
    merged_positions = np.concatenate([choline, positions[:, cl], positions[:, urea]], axis=1)
    merged_species = np.repeat([1, 2, 3], [ch1.sum(), cl.sum(), urea.sum()])
    return merged_positions, merged_species

if run_analysis:
    with active_directory(results_folder):
        with log_terminal("analysis_log.txt", "wt"):

            ## Partial RDFs of the new CG trajectory, on the same bins as the AA reference RDFs
            with np.load("cg_trajectory.npz") as cg_data:
                cg_positions = cg_data["position"][::rdf_stride]
                cg_cells = cg_data["cells"][::rdf_stride]
                cg_species = cg_data["species"][::rdf_stride]
                if mapping == "Martini":
                    cg_positions, cg_species = merge_choline_beads(cg_positions, cg_data["masses"][::rdf_stride], cg_cells, cg_species)
                _, _, cg_rdfs = calculate_rdf(
                    positions=cg_positions,
                    cutoff=22,  # 300 bins up to 22 Å reproduces the dataset rdf_bins
                    cells=cg_cells,
                    species=cg_species,
                    n_bins=300,
                )
            cg_rdfs = {key.replace("-", "_"): value for key, value in cg_rdfs.items()}
            np.savez("cg_rdfs.npz", rdf_bins=rdf_bins, **cg_rdfs)

if run_plotting:
    with active_directory(results_folder):
        with log_terminal("plotting_log.txt", "wt"):

            with np.load("cg_rdfs.npz") as saved_rdfs:
                cg_rdfs = {key: saved_rdfs[key] for key in saved_rdfs.files if key.startswith("rdf_values_")}

            with np.load(paper_cg_file) as paper_cg_data:
                paper_cg_rdfs = {key: paper_cg_data[key] for key in ref_rdfs}

            def rdf_error(cg_rdf, aa_rdf):
                return (rdf_bins[1] - rdf_bins[0]) * np.sum((cg_rdf - aa_rdf) ** 2)

            print("RDF error vs AA, dr * sum (g_CG - g_AA)^2 (Å)")
            print(f"{'pair':>10s} {'this run':>10s} {'paper CG':>10s}")
            for i, j in itertools.combinations_with_replacement(sorted(analysis_species_names), 2):
                key = f"rdf_values_{i}_{j}"
                pair_name = f"{analysis_species_names[i]}-{analysis_species_names[j]}"
                print(f"{pair_name:>10s} {rdf_error(cg_rdfs[key], ref_rdfs[key]):10.4f} {rdf_error(paper_cg_rdfs[key], ref_rdfs[key]):10.4f}")

                fig, ax = plt.subplots(figsize=(4.6, 3.4))
                ax.plot(rdf_bins, ref_rdfs[key], label="AA", color=AA_COLOR, linewidth=2)
                ax.plot(rdf_bins, paper_cg_rdfs[key], label="CG (paper)", color=CG_PAPER_COLOR, linestyle="--", linewidth=2)
                ax.plot(rdf_bins, cg_rdfs[key], label="CG (this run)", color=CG_RUN_COLOR, linestyle="--", linewidth=2)
                ax.set_title(f"{system} {temperature} K, {mapping} mapping: {pair_name}", fontsize=12)
                ax.set_xlabel(r"$r$ ($\mathrm{\AA}$)", fontsize=11)
                ax.set_ylabel(r"$g(r)$ (-)", fontsize=11)
                ax.tick_params(labelsize=10)
                ax.set_xlim((3.1, 12.9))
                ax.legend(fontsize=9, loc="upper right")
                ax.text(0.97, 0.06, f"SSE: {rdf_error(cg_rdfs[key], ref_rdfs[key]):.2f} " + r"$\mathrm{\AA}$",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
                fig.tight_layout()
                fig.savefig(f"rdf_{analysis_species_names[i]}_{analysis_species_names[j]}.png", dpi=150)
                plt.close(fig)
            print(f"RDF plots saved in {os.path.abspath(os.curdir)}")
