import numpy as np
import torch
import ase
import time
from tqdm import trange

import ase.build
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from hippynn.graphs import physics, inputs
from hippynn.graphs.gops import swap_pairfinders
from hippynn.graphs import Predictor
from hippynn.graphs.nodes.pairs import KDTreePairsMemory
from hippynn.experiment import load_checkpoint_from_cwd
from hippynn.tools import active_directory
from hippynn.molecular_dynamics.md import (
    Variable,
    NullUpdater,
    VelocityVerlet,
    MolecularDynamics,
)

from conftest import skip_if_no_models, ignore_relocation, ignore_weights_only_warning, ignore_sensitivity_warning, MODEL_DIR

from ase import Atoms
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)


ANI_MODEL = "hip0_b512_int1_p5_GPU0_seed589961"


def generate_ase_results(hippynn_model_location, n_steps=3):
    from ase import Atoms
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from hippynn.interfaces.ase_interface import HippynnCalculator

    a = 16.0  # Å box length
    symbols = "CCCHHHNO"

    positions = np.array(
        [
            [2.0, 2.0, 2.0],  # C1
            [3.5, 2.2, 2.1],  # C2 (bonded to C1)
            [5.0, 2.5, 2.2],  # C3 (bonded to C2)
            [1.2, 2.8, 2.1],  # H1 (near C1)
            [3.7, 1.1, 2.8],  # H2 (near C2)
            [5.5, 3.5, 1.8],  # H3 (near C3)
            [6.2, 2.9, 2.4],  # N (bonded-ish to C3)
            [7.5, 3.1, 3.0],  # O (near N)
        ]
    )

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[a, a, a],
        pbc=True,
    )

    rng = np.random.RandomState(1993)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=rng)

    with active_directory(hippynn_model_location, create=False):
        check = load_checkpoint_from_cwd(map_location="cpu")

    model = check["training_modules"].model

    positions_node = model.unique_node_from_name("R")
    energy_node = model.unique_node_from_name("HEnergy.mol_energy")
    force_node = physics.GradientNode("forces", (energy_node, positions_node), sign=-1)
    force_node.db_name = "forces"

    calc = HippynnCalculator(energy_node, en_unit=units.kcal / units.mol)
    calc.to(torch.float64)
    atoms.calc = calc

    print(f"Number of atoms: {len(positions)}")
    print(f"Species:\n{atoms.get_atomic_numbers()}")
    print(f"Masses:\n{atoms.get_masses()}")
    print(f"Cell:\n{atoms.cell.array}")
    print(f"Initial positions:\n{atoms.get_positions()}")
    print(f"Initial velocities:\n{atoms.get_velocities()}")
    print(f"Initial forces:\n{atoms.get_forces()}")
    print(f"Initial accelerations:\n{atoms.get_forces() / atoms.get_masses()[:, None]}")

    from ase.md.verlet import VelocityVerlet as aseVelocityVerlet

    dyn = aseVelocityVerlet(atoms, 1 * units.fs)
    md_pos = []
    md_vel = []
    md_force = []
    for i in range(n_steps):
        dyn.run(1)
        md_pos.append(atoms.get_positions())
        md_vel.append(atoms.get_velocities())
        md_force.append(atoms.get_forces())

    print(f"-----MD results-----")
    print(f"Number of steps: {n_steps}")
    print(f"Positions:\n{np.array(md_pos)}")
    print(f"Velocities:\n{np.array(md_vel)}")
    print(f"Forces:\n{np.array(md_force)}")

    return


@ignore_weights_only_warning
@ignore_relocation
@ignore_sensitivity_warning
@skip_if_no_models
def test_md_verlet():
    # Define coordinates and results from running the function `generate_ase_results` from above

    # fmt: off ##
    species = torch.tensor(
        [6, 6, 6, 1, 1, 1, 7, 8],
        dtype=int,
    ).unsqueeze(0) # add a batch axis

    masses = torch.tensor(
        [12.011, 12.011, 12.011, 1.008, 1.008, 1.008, 14.007, 15.999],
        dtype=torch.float64,
    ).unsqueeze(0) # add a batch axis

    # fmt: off ## Formatters tend to make these matrices hard to read.
    cell = torch.tensor(
        [[16.,  0.,  0.,],
         [ 0., 16.,  0.,],
         [ 0.,  0., 16.,]],
        dtype=torch.float64,
    ).unsqueeze(0) # add a batch axis

    init_positions = torch.tensor(
        [[2.0, 2.0, 2.0],   
         [3.5, 2.2, 2.1],   
         [5.0, 2.5, 2.2],   
         [1.2, 2.8, 2.1],   
         [3.7, 1.1, 2.8],   
         [5.5, 3.5, 1.8],   
         [6.2, 2.9, 2.4],   
         [7.5, 3.1, 3.0]], 
        dtype=torch.float64,
    ).unsqueeze(0) # add a batch axis

    init_velocities = torch.tensor(
        [[-0.00822128, -0.01170768, -0.00938906,],
         [ 0.0239996,  -0.02384575, -0.02124534,],
         [ 0.08267665,  0.00261897, -0.05979661,],
         [ 0.0463729,   0.01137487,  0.08822831,],
         [ 0.04209134,  0.18889452, -0.15332616,],
         [ 0.02099581, -0.15164287, -0.23867406,],
         [ 0.02223526,  0.01192979,  0.12204509,],
         [ 0.03641274,  0.0161685,  -0.03055966,]],
        dtype=torch.float64,
    ).unsqueeze(0) # add a batch axis
    
    init_acc = torch.tensor(
        [[ 0.79568028,  0.32844587,  0.07309023],
         [-0.14169344, -0.39031643,  0.2426869 ],
         [-0.77908242, -0.20590222, -0.0839564 ],
         [ 0.66706587, -1.8328339,  -0.31282832],
         [-1.54045893,  3.49569841, -2.48280871],
         [ 1.02746512, -0.21992615,  0.54266946],
         [ 0.5641636,   0.19546156,  0.17868188],
         [-0.40971443, -0.06100963, -0.18852456]],
        dtype=torch.float64,
    ).unsqueeze(0) # add a batch axis

    result_pos = np.array(
        [[[2.00303102, 2.0004345,  1.99943035,],
          [3.50167384, 2.19577471, 2.09908392,],
          [5.00436257, 2.49926393, 2.19372133,],
          [1.20777317, 2.79227524, 2.10715723,],
          [3.69670292, 1.13541871, 2.77296151,],
          [5.50701912, 3.4840436,  1.77917376,],
          [6.20490578, 2.90211479, 2.41285013,],
          [7.50160014, 3.10129386, 2.99608872,]],
 
         [[2.01378926, 2.00380454, 1.99961819,],
          [3.50203539, 2.18790628, 2.10043831,],
          [5.00149586, 2.49673513, 2.18656789,],
          [1.22104457, 2.76801967, 2.11136903,],
          [3.67630735, 1.20398215, 2.72206249,],
          [5.5230105,  3.46527314, 1.76388369,],
          [6.21508362, 2.90604868, 2.42735465,],
          [7.49936158, 3.10200273, 2.99045265,]],
 
         [[2.0320744,  2.00935572, 2.0006753, ],
          [3.50156638, 2.17746924, 2.10348298,],
          [4.99256866, 2.49285478, 2.17860319,],
          [1.2363625,  2.73120472, 2.11313041,],
          [3.63380054, 1.29664171, 2.65208867,],
          [5.54492426, 3.4440373,  1.75293322,],
          [6.22972448, 2.91144304, 2.44335003,],
          [7.49362933, 3.102167,   2.9832816, ]]],
        dtype=np.float64,
    )

    result_vel = np.array(
        [[[ 7.01908192e-02,  1.93660617e-02, -1.94350754e-03,],
          [ 1.03606355e-02, -6.15600893e-02,  2.23110007e-03,],
          [ 7.61428402e-03, -1.66190174e-02, -6.83728192e-02,],
          [ 1.07122202e-01, -1.62787985e-01,  5.78712460e-02,],
          [-1.20601582e-01,  5.29295442e-01, -3.96721606e-01,],
          [ 1.17129285e-01, -1.76768488e-01, -1.83841154e-01,],
          [ 7.67794570e-02,  3.07893324e-02,  1.39242063e-01,],
          [-3.24974301e-03,  1.01944154e-02, -4.85984279e-02,]],
 
         [[ 1.47838113e-01,  4.54112442e-02,  6.33711543e-03,],
          [-5.46981187e-04, -9.31794866e-02,  2.23922959e-02,],
          [-6.00340175e-02, -3.26241903e-02, -7.69551803e-02,],
          [ 1.45526902e-01, -3.10864374e-01,  3.04050012e-02,],
          [-3.20189024e-01,  8.20665818e-01, -6.15273297e-01,],
          [ 1.92946723e-01, -2.03642207e-01, -1.33570976e-01,],
          [ 1.26333475e-01,  4.74831938e-02,  1.55252209e-01,],
          [-4.05734147e-02,  4.44452356e-03, -6.51914856e-02,]],
 
         [[ 2.19679261e-01,  6.28986877e-02,  1.49832144e-02,],
          [-4.35827569e-03, -1.09282937e-01,  3.42841969e-02,],
          [-1.11781063e-01, -4.31618666e-02, -8.41175288e-02,],
          [ 1.39449759e-01, -4.09586341e-01,  9.62583635e-03,],
          [-5.81283790e-01,  9.49872962e-01, -7.47588201e-01,],
          [ 2.35437671e-01, -2.21860320e-01, -1.00836249e-01,],
          [ 1.65507106e-01,  6.03569577e-02,  1.68425719e-01,],
          [-7.29379030e-02, -7.27244992e-04, -7.91832402e-02,]]],
        dtype=np.float64,
    )

    result_forces = np.array(
        [[[ 9.61924248,  3.65430954,  0.94296853,],
          [-1.63361241, -4.53518164,  2.82639486,],
          [-8.9994004,  -2.23167505, -1.08896439,],
          [ 0.57441015, -1.72700432, -0.30771442,],
          [-1.78631066,  3.46269031, -2.49275258,],
          [ 0.93734893, -0.29399012,  0.57837417,],
          [ 7.65358409,  2.6408498,   2.4017224, ],
          [-6.36526218, -0.96999851, -2.86002857,]],

         [[ 9.36987756,  2.71519891,  1.08210839,],
          [-1.03391186, -3.19753508,  2.10414871,],
          [-7.54440488, -1.68248774, -1.00990443,],
          [ 0.21380403, -1.31210061, -0.25600003,],
          [-2.31000192,  2.51736598, -1.99278035,],
          [ 0.61872054, -0.25756339,  0.45336593,],
          [ 6.47905766,  2.12018415,  2.16431833,],
          [-5.79314113, -0.90306224, -2.54525654,]],

         [[ 8.1993139,   1.56146219,  1.03234793,],
          [ 0.10183651, -0.74066202,  0.80408829,],
          [-5.11065108, -0.89456536, -0.74169164,],
          [-0.33853071, -0.7140591,  -0.17046946,],
          [-3.04868081,  0.13446845, -0.72283742,],
          [ 0.25335942, -0.11634334,  0.21847832,],
          [ 4.69313207,  1.55137064,  1.59272294,],
          [-4.74977931, -0.78167147, -2.01263895,]]],
        dtype=np.float64,
    ) / (units.kcal / units.mol) # ASE returns everything in its standard units

    # fmt: on ## END matrices.

    # Load and prepare model
    with active_directory(MODEL_DIR / ANI_MODEL, create=False):
        check = load_checkpoint_from_cwd(map_location="cpu")

    model = check["training_modules"].model

    positions_node = model.unique_node_from_name("R")
    energy_node = model.unique_node_from_name("HEnergy.mol_energy")
    force_node = physics.GradientNode("forces", (energy_node, positions_node), sign=-1)
    force_node.db_name = "forces"

    pair_indexer_node = model.unique_node_from_name("PairIndexer")
    cutoff = pair_indexer_node.torch_module.hard_dist_cutoff

    cell_node = inputs.CellNode(name="C", db_name="C")

    swap_pairfinders(positions_node, KDTreePairsMemory, cell_node=cell_node, module_kwargs={"skin": 1.0, "dist_hard_max": cutoff})

    model = Predictor(inputs=[*model.input_nodes, cell_node], outputs=[force_node])
    model.to(torch.float64)

    # Set up MD

    position_variable = Variable(
        name="position",
        data={
            "position": init_positions,
            "velocity": init_velocities,
            "acceleration": init_acc,
            "mass": masses,
            "cell": cell,  # Optional. If added, coordinates will be wrapped in each step of the VelocityVerlet updater. Otherwise, they will be temporarily wrapped for model evaluation only and stored in their unwrapped form
        },
        model_input_map={
            "R": "position",
        },
        updater=VelocityVerlet(
            force_db_name="forces",
            time_units=1,  # this will use the ASE default time unit
            # position_units = units.Bohr,
            force_units=(units.kcal / units.mol),
        ),
    )

    species_variable = Variable(
        name="species",
        data={"species": species},
        model_input_map={"Z": "species"},
        updater=NullUpdater(),
    )

    cell_variable = Variable(
        name="cell",
        data={"cell": cell},
        model_input_map={"C": "cell"},
        updater=NullUpdater(),
    )

    emdee = MolecularDynamics(
        variables=[position_variable, species_variable, cell_variable],
        model=model,
    )

    emdee.run(dt=1 * units.fs, n_steps=3, record_every=1)

    data = emdee.get_data()

    assert np.allclose(data["position_position"][:, 0], result_pos)
    assert np.allclose(data["position_velocity"][:, 0], result_vel)
    assert np.allclose(data["position_force"][:, 0], result_forces)

    return


if __name__ == "__main__":
    generate_ase_results(MODEL_DIR / ANI_MODEL)
