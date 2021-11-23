import torch
from seqm.basics import Energy, Force, parameterlist
from seqm.seqm_functions.constants import Constants
from seqm.seqm_functions.parameters import params
import os
from hippynn.graphs.nodes.base import MultiNode, AutoKw, find_unique_relative, ExpandParents, SingleNode
from hippynn.graphs.indextypes import IdxType
from hippynn.graphs.nodes.inputs import PositionsNode, SpeciesNode


class MLSEQM(torch.nn.Module):
    def __init__(self, seqm_parameters):
        super().__init__()
        self.learned = seqm_parameters["learned"]  # parameters for each real atom
        seqm_parameters["Hf_flag"] = False  # true: Hf, false: Etot-Eiso
        self.energy = Energy(seqm_parameters)
        self.const = Constants()
        self.elements = seqm_parameters["elements"]
        self.method = seqm_parameters["method"]
        self.filedir = seqm_parameters["parameter_file_dir"]
        self.n_output_parameters = len(self.learned)
        self.p = torch.nn.Parameter(
            params(method=self.method, elements=self.elements, root_dir=self.filedir, parameters=self.learned),
            requires_grad=True,
        )
        #
        self.const = Constants()
        self.save("./MOPAC/")  # save the initial value
        """
        #if only training to energy related terms:
        seqm_parameters['eig'] = False # don't need orbital energy
        seqm_parameters['scf_backward'] = 0  # ignore gradient on density matrix
        """

    def save(self, fdir="./MLSEQM/"):
        if not os.path.exists(fdir):
            os.system("mkdir %s" % fdir)
        fn = fdir + "parameters_" + self.method + "_MOPAC.csv"
        additional_parameters = [x for x in parameterlist[self.method] if x not in self.learned]
        p1 = params(method=self.method, elements=self.elements, root_dir=self.filedir, parameters=additional_parameters)
        with open(fn, "w") as f:
            f.write("N, " + ", ".join(["%23s" % s for s in self.learned]))
            if additional_parameters:
                f.write(", " + ", ".join(["%23s" % s for s in additional_parameters]))
            f.write("\n")
            for ele in self.elements:
                if ele == 0:
                    continue
                f.write(str(ele) + ", ")
                f.write(", ".join(["%23.16e" % x for x in self.p[ele]]))
                if additional_parameters:
                    f.write(", " + ", ".join(["%23.16e" % x for x in p1[ele]]))
                f.write("\n")

    def forward(self, coordinates, species):
        cond = species.reshape(-1) > 0
        Z = species.reshape(-1)[cond]
        learned_parameters = dict()
        for i in range(self.n_output_parameters):
            learned_parameters[self.learned[i]] = self.p[Z, i]

        Etot_m_Eiso, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, notconverged = self.energy(
            self.const, coordinates, species, learned_parameters=learned_parameters, all_terms=True
        )
        n_molecule, n_atom = species.shape
        atomic_charge = self.const.tore[species] - P.diagonal(dim1=1, dim2=2).reshape(n_molecule, n_atom, -1).sum(dim=2)

        return (
            Etot.reshape(-1, 1),
            Etot_m_Eiso.reshape(-1, 1),
            e,
            P,
            Eelec.reshape(-1, 1),
            Enuc.reshape(-1, 1),
            Eiso.reshape(-1, 1),
            charge,
            notconverged.reshape(-1, 1),
            atomic_charge,
        )


class MLSEQM_Node(AutoKw, MultiNode):
    _input_names = "Positions", "Species"
    _output_names = (
        "mol_energy",
        "Etot_m_Eiso",
        "orbital_energies",
        "single_particle_density_matrix",
        "electric_energy",
        "nuclear_energy",
        "isolated_atom_energy",
        "orbital_charges",
        "notconverged",
        "atomic_charge",
    )
    _main_output = "mol_energy"
    _output_index_states = (IdxType.Molecules,) * len(_output_names)
    _auto_module_class = MLSEQM

    def __init__(self, name, parents, seqm_parameters, module="auto", **kwargs):
        self.module_kwargs = dict(seqm_parameters=seqm_parameters)
        super().__init__(name, parents, module=module, **kwargs)
