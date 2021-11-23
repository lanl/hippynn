import seqm
import torch
from seqm.seqm_functions.fock import fock
from seqm.seqm_functions.hcore import hcore
from seqm.seqm_functions.diag import sym_eig_trunc1
from seqm.basics import Hamiltonian, Energy


class Hamiltonian_One(Hamiltonian):
    def __init__(self, seqm_parameters):
        super().__init__(seqm_parameters)

    def forward(
        self,
        const,
        molsize,
        nHeavy,
        nHydro,
        nOccMO,
        Z,
        maskd,
        mask,
        atom_molid,
        pair_molid,
        idxi,
        idxj,
        ni,
        nj,
        xij,
        rij,
        parameters,
        P0,
    ):
        beta = torch.cat((parameters["beta_s"].unsqueeze(1), parameters["beta_p"].unsqueeze(1)), dim=1)
        nmol = nOccMO.shape[0]
        zetas = parameters["zeta_s"]
        zetap = parameters["zeta_p"]
        uss = parameters["U_ss"]
        upp = parameters["U_pp"]
        gss = parameters["g_ss"]
        gsp = parameters["g_sp"]
        gpp = parameters["g_pp"]
        gp2 = parameters["g_p2"]
        hsp = parameters["h_sp"]
        M, w = hcore(
            const,
            nmol,
            molsize,
            maskd,
            mask,
            idxi,
            idxj,
            ni,
            nj,
            xij,
            rij,
            Z,
            zetas,
            zetap,
            uss,
            upp,
            gss,
            gpp,
            gp2,
            hsp,
            beta,
        )
        F = fock(nmol, molsize, P0, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
        Hcore = M.reshape(nmol, molsize, molsize, 4, 4).transpose(2, 3).reshape(nmol, 4 * molsize, 4 * molsize)
        if self.eig:
            e, v = sym_eig_trunc1(F, nHeavy, nHydro, nOccMO, eig_only=True)
            return F, e, P0, Hcore, w, None, None
        else:
            return F, None, P0, Hcore, w, None, None


class Energy_One(Energy):
    def __init__(self, seqm_parameters):
        super().__init__(seqm_parameters)
        self.hamiltonian = Hamiltonian_One(seqm_parameters)


from hippynn.interfaces.pyseqm_interface.seqm_modules import SEQM_Energy, SEQM_All, pack_par


class SEQM_One_Energy(SEQM_Energy):
    def __init__(self, seqm_parameters):
        super().__init__(seqm_parameters)
        self.energy = Energy_One(seqm_parameters)

    def forward(self, par_atom, coordinates, species, single_particle_density_matrix):
        learned_parameters = pack_par(self, species, par_atom)

        Etot_m_Eiso, Etot = self.energy(
            self.const,
            coordinates,
            species,
            learned_parameters=learned_parameters,
            all_terms=True,
            P0=single_particle_density_matrix,
        )[:2]
        return Etot.reshape(-1, 1), Etot_m_Eiso.reshape(-1, 1)


class SEQM_One_All(SEQM_All):
    def __init__(self, seqm_parameters):
        super().__init__(seqm_parameters)
        self.energy = Energy_One(seqm_parameters)

    def forward(self, par_atom, coordinates, species, single_particle_density_matrix):
        learned_parameters = pack_par(self, species, par_atom)

        Etot_m_Eiso, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, notconverged = self.energy(
            self.const,
            coordinates,
            species,
            learned_parameters=learned_parameters,
            all_terms=True,
            P0=single_particle_density_matrix,
        )

        n_molecule, n_atom = species.shape
        atomic_charge = self.const.tore[species] - P.diagonal(dim1=1, dim2=2).reshape(n_molecule, n_atom, -1).sum(dim=2)

        return (
            Etot.reshape(-1, 1),
            Etot_m_Eiso.reshape(-1, 1),
            e,
            Eelec.reshape(-1, 1),
            Enuc.reshape(-1, 1),
            Eiso.reshape(-1, 1),
            atomic_charge,
        )


from hippynn.graphs.nodes.networks import Network
from hippynn.graphs.nodes.targets import HChargeNode  # , HBondNode
from hippynn.graphs.nodes.inputs import PositionsNode, SpeciesNode
from hippynn.graphs.nodes.base import InputNode
from hippynn.graphs.indextypes import IdxType
from hippynn.graphs.nodes.base import MultiNode, AutoKw, find_unique_relative, ExpandParents, SingleNode


class NotConvergedNode(InputNode):
    _index_state = IdxType.Molecules
    input_type_str = "notconverged"


class DensityMatrixNode(InputNode):
    _index_state = IdxType.Molecules
    input_type_str = "single_particle_density_matrix"


class SEQM_One_EnergyNode(ExpandParents, AutoKw, MultiNode):
    _input_names = "par_atom", "Positions", "Species", "single_particle_density_matrix"
    _output_names = "mol_energy", "Etot_m_Eiso"
    _main_output = "Etot_m_Eiso"
    _output_index_states = (IdxType.Molecules,) * len(_output_names)
    _auto_module_class = SEQM_One_Energy

    @_parent_expander.match(Network, DensityMatrixNode)
    def expand0(self, network, single_particle_density_matrix, seqm_parameters, decay_factor=1.0e-2, **kwargs):

        n_target_peratom = len(seqm_parameters["learned"])

        par_atom = HChargeNode(
            "SEQM_Atom_Params", network, module_kwargs=dict(n_target=n_target_peratom, first_is_interacting=True)
        )

        with torch.no_grad():
            for layer in par_atom.torch_module.layers:
                layer.weight.data *= decay_factor
                layer.bias.data *= decay_factor

        positions = find_unique_relative(network, PositionsNode)
        species = find_unique_relative(network, SpeciesNode)

        return par_atom.main_output, positions, species, single_particle_density_matrix

    _parent_expander.assertlen(4)
    _parent_expander.get_main_outputs()
    _parent_expander.require_idx_states(IdxType.Atoms, None, None, None)

    def __init__(self, name, parents, seqm_parameters, decay_factor=1.0e-2, module="auto", **kwargs):
        parents = self.expand_parents(parents, seqm_parameters=seqm_parameters, decay_factor=decay_factor, **kwargs)
        self.module_kwargs = dict(seqm_parameters=seqm_parameters)
        super().__init__(name, parents, module=module, **kwargs)


class SEQM_One_AllNode(SEQM_One_EnergyNode):
    _input_names = "par_atom", "Positions", "Species", "single_particle_density_matrix"
    _output_names = (
        "mol_energy",
        "Etot_m_Eiso",
        "orbital_energies",
        "electric_energy",
        "nuclear_energy",
        "isolated_atom_energy",
        "atomic_charge",
    )
    _main_output = "Etot_m_Eiso"
    _output_index_states = (IdxType.Molecules,) * len(_output_names)
    _auto_module_class = SEQM_One_All
