import torch
from seqm.basics import Energy
from seqm.seqm_functions.constants import Constants
from seqm.seqm_functions.parameters import params
from seqm.seqm_functions.energy import elec_energy_isolated_atom

from .check import check, check_dist


class Scale(torch.nn.Module):
    def __init__(self, func=torch.sqrt):
        super().__init__()
        self.func = func

    def forward(self, notconverged):
        frac = 1.0 - notconverged.sum().double() / (notconverged.shape[0] + 0.0)
        return self.func(frac.detach())


class SEQM_MolMask(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, notconverged):
        mask = ~(notconverged.detach())
        return mask


class AtomMask(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, species):
        return species > 0


def num_orb(species, target_method="wB97X"):
    tore = Constants().tore.to(species.device)
    nHeavy = torch.sum(species > 1, dim=1)
    nHydro = torch.sum(species == 1, dim=1)

    # MNDO information
    nocc = (torch.sum(tore[species], dim=1) / 2.0).reshape(-1).type(torch.int64)
    norb = 4 * nHeavy + nHydro
    # nvirt = norb-nocc

    if target_method == "wB97X":
        # wB97X information
        # nocc0 = (torch.sum(species,dim=1)/2.0).type(torch.int64)
        # norb0 = torch.sum(species>0,dim=1)*4
        nvirt0 = torch.sum(species > 0, dim=1) * 4 - (torch.sum(species, dim=1) / 2.0).type(torch.int64)
        return torch.min(norb, nocc + nvirt0), nocc
    elif target_method == "gaussian":
        nvirt0 = (
            torch.sum(species > 1, dim=1) * 15
            + torch.sum(species == 1, dim=1) * 2
            - (torch.sum(species, dim=1) / 2.0).type(torch.int64)
        )
        return torch.min(norb, nocc + nvirt0), nocc
    else:
        raise KeyError(
            "Computing orbital number \
                        information is not implemented yet for ",
            target_method,
        )


"""
assume the target dataset is preprocessed such that
the data there has same number of occupied orbitals and at most the number of virtual orbitals
from SEQM.
as for certain molecules like O3, SEQM may have more virtual orbitals than from wB97X
For this case, the code here will trim the virtual orbitals from SEQM to match
"""


class SEQM_OrbitalMask(torch.nn.Module):
    def __init__(self, target_method, nOccVirt=None):
        """
        nOccVirt : if None, choose all the orbitals which MNDO and target methods share
                   else [NOCC, NVIRT], i.e. choose highest NOCC occupied orbtials (HOMO-NOCC+1, HOMO-NOCC+2, ..., HOMO)
                                            and lowest NVIRT virtual orbitals (LUMO, LUMO+1, ..., LUMO+NVIRT-1)
                        if there are not sufficient number of virtual/occupied orbitals, just choose whatever they have
                        i.e. NOCC <= nocc, NVIRT <= norb - nocc
        """
        super().__init__()
        self.target_method = target_method
        self.nOccVirt = nOccVirt

    def forward(self, species):
        norb, nocc = num_orb(species, self.target_method)
        nmol, molsize = species.shape
        tmp = (
            torch.arange(4 * molsize, dtype=torch.int64, device=species.device).reshape(1, -1).expand(nmol, 4 * molsize)
        )
        if isinstance(self.nOccVirt, (list, tuple)):
            NOCC, NVIRT = self.nOccVirt
            nindx1 = nocc - NOCC
            nindx1[nindx1 < 0] = 0
            nindx2 = nocc + NVIRT
            nindx2 = torch.min(norb, nindx2)

            mask = (tmp >= nindx1.reshape(-1, 1)) * (tmp < nindx2.reshape(-1, 1))
        else:
            mask = tmp < norb.reshape(-1, 1)

        return mask


class SEQM_MaskOnMol(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, var, mol_mask):
        if var.dim() == 1:
            return var.unsqueeze(1)[mol_mask.reshape(-1)]
        else:
            return var[mol_mask.reshape(-1)]


class SEQM_MaskOnMolAtom(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, var, mol_mask, atom_mask):
        # mol_mask shape : (mol, 1)
        # atom_mask shape : (mol, atom)
        mask = mol_mask * atom_mask
        return var[mask]


class SEQM_MaskOnMolOrbital(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, var, mol_mask, orbital_mask):
        # mol_mask shape : (mol, 1)
        # orbital_mask shape : (mol, orbital)
        # var shape : (mol, orbital)
        mask = mol_mask * orbital_mask
        return var[mask]


class SEQM_MaskOnMolOrbitalAtom(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, var, mol_mask, orbital_mask, atom_mask):
        # mol_mask shape : (mol, 1)
        # orbital_mask shape : (mol, orbital)
        # atom_mask shape : (mol, atom)
        # var shape : (mol, orbital, atom)
        mask = mol_mask.unsqueeze(2) * orbital_mask.unsqueeze(2) * atom_mask.unsqueeze(1)
        return var[mask]


def pack_par(obj, species, par_atom, par_bond=None):

    learned_parameters = {}
    eps = 1.0e-2
    if torch.is_tensor(par_bond):
        learned_parameters["Kbeta"] = par_bond + 1.0

    cond = species.reshape(-1) > 0
    Z = species.reshape(-1)[cond]

    for i in range(obj.n_output_parameters):
        # learned_parameters[obj.learned[i]] = par_atom[cond,i]*obj.weight[i]+obj.p[Z,i]
        learned_parameters[obj.learned[i]] = par_atom[:, i] * obj.weight[i] + obj.p[Z, i]

    for x in ["zeta_s", "zeta_p", "g_ss", "h_sp", "alpha", "Gaussian1_L", "Gaussian2_L", "Gaussian3_L", "Gaussian4_L"]:
        if x in obj.learned:
            learned_parameters[x] = obj.softplus(learned_parameters[x]) + eps

    assert ("g_pp" in obj.learned and "g_p2" in obj.learned) or (
        "g_pp" not in obj.learned and "g_p2" not in obj.learned
    ), "For convinience, both of g_pp and g_p2 should be provided from HIPNN or use constant ones"

    # hpp = 0.5*(gpp-gp2)>0
    if "g_pp" in obj.learned and "g_p2" in obj.learned:
        learned_parameters["g_pp"] = (
            obj.softplus(learned_parameters["g_pp"] - learned_parameters["g_p2"]) + learned_parameters["g_p2"] + eps
        )

    if obj.not_learned:
        # not_learned is a dict, e.g. : {1:['U_pp','h_sp']}, means use constant parameters for Hydrogen in U_pp and h_sp
        for k in obj.not_learned:
            for x in obj.not_learned[k]:
                if x in obj.learned:
                    i = obj.learned.index(x)
                    learned_parameters[x][Z == k] = obj.p[k, i]

    return learned_parameters


class SEQM_Energy(torch.nn.Module):
    def __init__(self, seqm_parameters):
        """
        Constructor
        """
        super().__init__()
        # ignore gradient on density matrix, based on Hellmann-Feynman Theorm
        # only works if targets are total energy or heat of formation
        # ?? for heat of formation, Hf = Etot - Eiso + Eexp
        # Eiso depends on parameters
        self.hartree_eV = 27.2113834
        self.learned = seqm_parameters["learned"]  # parameters for each real atom
        self.n_output_parameters = len(self.learned)
        seqm_parameters["eig"] = False  # don't need orbital energy
        seqm_parameters["scf_backward"] = 0  # ignore gradient on density matrix ``
        seqm_parameters["Hf_flag"] = False  # true: Hf, false: Etot-Eiso
        self.energy = Energy(seqm_parameters)
        self.const = Constants()
        self.elements = seqm_parameters["elements"]
        self.method = seqm_parameters["method"]
        self.filedir = seqm_parameters["parameter_file_dir"]
        self.p = params(method=self.method, elements=self.elements, root_dir=self.filedir, parameters=self.learned)
        self.softplus = torch.nn.Softplus(beta=5.0)
        self.weight = []
        for term in self.learned:
            if "zeta" in term or "alpha" in term or "Gaussian" in term:
                self.weight.append(1.0)
            else:
                self.weight.append(self.hartree_eV)
        self.weight = torch.nn.Parameter(torch.tensor(self.weight), requires_grad=False)
        self.not_learned = seqm_parameters.get("not_learned", None)

    def forward(self, par_atom, coordinates, species):
        """
        get the energy terms
        """
        learned_parameters = pack_par(self, species, par_atom)

        Etot_m_Eiso, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, notconverged = self.energy(
            self.const, coordinates, species, learned_parameters=learned_parameters, all_terms=True
        )

        return Etot.reshape(-1, 1), Etot_m_Eiso.reshape(-1, 1), notconverged.reshape(-1, 1)


class SEQM_All(torch.nn.Module):
    def __init__(self, seqm_parameters):
        """
        Constructor
        """
        super().__init__()
        self.hartree_eV = 27.2113834
        self.learned = seqm_parameters["learned"]  # parameters for each real atom
        self.n_output_parameters = len(self.learned)
        seqm_parameters["eig"] = True  # don't need orbital energy
        if "scf_backward" not in seqm_parameters:
            seqm_parameters["scf_backward"] = 1  # ignore gradient on density matrix ``
        elif seqm_parameters["scf_backward"] == 0:
            raise ValueError("scf_backward must be 1 or 2 for training with gradient on density matrix")
        seqm_parameters["Hf_flag"] = False  # true: Hf, false: Etot-Eiso
        self.energy = Energy(seqm_parameters)
        self.const = Constants()
        self.elements = seqm_parameters["elements"]
        self.method = seqm_parameters["method"]
        self.filedir = seqm_parameters["parameter_file_dir"]
        self.p = params(method=self.method, elements=self.elements, root_dir=self.filedir, parameters=self.learned)
        self.softplus = torch.nn.Softplus(beta=5.0)
        self.weight = []
        for term in self.learned:
            if "zeta" in term or "alpha" in term or "Gaussian" in term:
                self.weight.append(1.0)
            else:
                self.weight.append(self.hartree_eV)
        self.weight = torch.nn.Parameter(torch.tensor(self.weight), requires_grad=False)
        self.not_learned = seqm_parameters.get("not_learned", None)

    def forward(self, par_atom, coordinates, species):
        """
        get the energy terms
        """
        learned_parameters = pack_par(self, species, par_atom)

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
