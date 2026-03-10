"""
Layers for physical operations
"""
import warnings

import torch

from .. import indexers, pairs


class CoulombEnergy(torch.nn.Module):
    """ Computes the Coulomb Energy of the molecule/configuration. 
    
    Coulomb energies is defined for pairs of atoms. Here, we adopt the 
    convention that the Coulomby energy for a pair of atoms is evenly
    partitioned to both atoms as the 'per-atom energies'. Therefore, the 
    atom energies sum to the molecular energy; similar to the HEnergy. 
    """
    def __init__(self, energy_conversion_factor):
        super().__init__()
        self.register_buffer("energy_conversion_factor", torch.tensor(energy_conversion_factor))
        self.summer = indexers.MolSummer()

    def forward(self, charges, pair_dist, pair_first, pair_second, system_index, n_systems):
        voltage_pairs = self.energy_conversion_factor * (charges[pair_second] / pair_dist.unsqueeze(1))
        n_atoms, _ = charges.shape
        voltage_atom = torch.zeros((n_atoms, 1), device=charges.device, dtype=charges.dtype)
        voltage_atom.index_add_(0, pair_first, voltage_pairs)
        coulomb_atoms = 0.5*voltage_atom * charges
        coulomb_molecule = self.summer(coulomb_atoms, system_index, n_systems)
        return coulomb_molecule, coulomb_atoms, voltage_atom


class ScreenedCoulombEnergy(CoulombEnergy):
    """ Computes the Coulomb Energy of the molecule/configuration. 
    
    The convention for the atom energies is the same as CoulombEnergy
    and the HEnergy. 
    """
    
    def __init__(self, energy_conversion_factor, screening, radius=None):
        super().__init__(energy_conversion_factor)
        if screening is None:
            raise ValueError("Screened Coulomb requires specification of a screening type.")
        if radius is None:
            raise ValueError("Screened Coulomb requires specification of a radius")

        if isinstance(screening, type):
            screening = screening()

        self.radius = radius

        self.screening = screening
        self.bond_summer = pairs.MolPairSummer()

    def forward(self, charges, pair_dist, pair_first, pair_second, system_index, n_systems):
        screening = self.screening(pair_dist, self.radius).unsqueeze(1)
        screening = torch.where((pair_dist < self.radius).unsqueeze(1), screening, torch.zeros_like(screening))

        # Voltage pairs for per-atom energy
        voltage_pairs = self.energy_conversion_factor * (charges[pair_second] / pair_dist.unsqueeze(1)) 
        voltage_pairs = voltage_pairs * screening 
        n_atoms, _ = charges.shape
        voltage_atom = torch.zeros((n_atoms, 1), device=charges.device, dtype=charges.dtype)
        voltage_atom.index_add_(0, pair_first, voltage_pairs) 
        coulomb_atoms = 0.5 * voltage_atom * charges
        coulomb_molecule = self.summer(coulomb_atoms, system_index, n_systems)

        return coulomb_molecule, coulomb_atoms, voltage_atom


class CombineScreenings(torch.nn.Module):
    """ Returns products of different screenings for Screened Coulomb Interactions.
    """
    def __init__(self, screening_list):
        super().__init__()
        self.SL = torch.nn.ModuleList(screening_list)

    def forward(self, pair_dist, radius):
        """ Product of different screenings applied to pair_dist upto radius.

        :param pair_dist: torch.tensor, dtype=float64: 'Neighborlist' distances for coulomb energies.
        :param radius: Maximum radius that Screened-Coulomb is evaluated upto.
        :return screening: Weights for screening for all pair_dist.
        """
        screening = None

        for s in self.SL:
            if screening is None:
                screening = s(pair_dist=pair_dist, radius=radius)
            else:
                screening = screening * s(pair_dist=pair_dist, radius=radius)

        return screening


class AlphaScreening(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha


# Note: This is somewhat incomplete as it does not include a k-space contribution -- more is needed
class EwaldRealSpaceScreening(AlphaScreening):
    def __init__(self, alpha):
        warnings.warn("Ewald implementation incomplete, does not include k-space contributions.")
        super().__init__(alpha)

    def forward(self, pair_dist, radius):
        q = pair_dist / radius
        eta = self.alpha * radius
        return torch.erfc(eta * q)


# Note: typically
class WolfScreening(AlphaScreening):
    def __init__(self, alpha):
        warnings.warn("Wolf implementation currently uses exact derivative of the potential.")
        super().__init__(alpha)

    def forward(self, pair_dist, radius):
        q = pair_dist / radius
        eta = self.alpha * radius
        return torch.erfc(eta * q) - q * torch.erfc(eta)


class LocalDampingCosine(AlphaScreening):
    """ Local damping using complement of the hipnn cutoff function. ('glue-on' method)
        g = 1 if pair_dist > R_cutoff, 1 - [cos(pi/2 * dist * R_cutoff)]^2  otherwise
    """
    def __init__(self, alpha): 
        """ 
        :param alpha: R_cutoff for glue-on function to ensure 
            smooth crossover from hipnn energy to long-range coulomb energy.  
        """
        super().__init__(alpha) 


    def forward(self, pair_dist, radius):
        """
        :param pair_dist: torch.tensor, dtype=float64: 'Neighborlist' distances for coulomb energies.
        :param radius: Maximum radius that Screened-Coulomb is evaluated upto. 
        :return screening: Weights for screening for each pair.
        """
        pi = torch.tensor([3.141592653589793238], device=pair_dist.device)        
        screening = torch.subtract(torch.tensor([1.0], device=pair_dist.device), torch.square(torch.cos(0.5*pi*pair_dist/self.alpha)))
    
        # pair_dist greater than cut-off; no local-damping. 
        screening = torch.where((pair_dist<self.alpha), screening, torch.ones_like(screening))
        
        return screening


class QScreening(torch.nn.Module):
    def __init__(self, p_value):
        super().__init__()
        self.p_value = p_value

    @property
    def p_value(self):
        return self._p_value

    @p_value.setter
    def p_value(self, value):
        value = int(value)
        self._p_value = value
        powers = torch.arange(1, value + 1, dtype=torch.long).unsqueeze(0)
        self.register_buffer("powers", powers)

    def forward(self, pair_dist, radius):
        q = pair_dist / radius
        q_factors = 1 - torch.pow(q.unsqueeze(1), self.powers)
        product = q_factors.prod(dim=1)
        return product



def change_tanh_range(vals, minn, maxx):
    '''
    Map [-1,1] to [min,  max]
    '''
    vals = (vals + 1.0) / 2.0 # [0,1]
    vals = vals * (maxx - minn) + minn
    return vals

class ChEQ(torch.nn.Module):
    def __init__(self, units={"energy": "eV", "length": "Angstrom"}, lower_bound=0.0):
        """
        default unit used here
        Length: Angstom
        Charge: elementary charge +e
        Energy: eV
        """
        super().__init__()

        self.units = units
        self.bound = lower_bound  # It was used to enforece Hubbard U to be positive (together with softplus), but it is replaced by using Tanh() here

        if self.units["length"] == "Angstrom":
            length_scale = 1.0
        elif self.units["length"] == "Bohr":
            length_scale = 0.529177210903
        else:
            raise ValueError("length with unit {} not supported yet".format(self.units["length"]))

        if self.units["energy"] == "eV":
            energy_scale = 1.0
        elif self.units["energy"] == "Hartree":
            energy_scale = 27.211386245988
        elif self.units["energy"] == "kcal/mol":
            energy_scale = 23.0609
        else:
            raise ValueError("energy with unit {} not supported yet".format(self.units["energy"]))

        self.conversion_factor = 1.0 / length_scale / energy_scale
        self.length_scale = length_scale
        self.energy_scale = energy_scale

        #self.c = self.bound * E_h  # It is no longer used

        self.tanh_chi = torch.nn.Tanh()

        self.tanh_U = torch.nn.Tanh()  # torch.nn.Softplus(beta=10.0) # used for enforce U to be positive, U: Hubbard_U

    def forward(self, species, coordinates, U, chi):
        """
        species : shape (n_molecule, n_atom)
        coordinates : shape (n_molecule, n_atom, 3), unit Angstrom or bohr
        U : Hubbard U, diagonal part of A, shape (n_real_atom, 1), output from HCharge Node
        chi : Electronegatvitiy, shape (n_real_atom, 1), output from HCharge Node
        """
        dtype = coordinates.dtype
        device = coordinates.device
        nonblank = species > 0
        n_molecule, n_atom = coordinates.shape[:2]
        
        # parameter ranges
        chi_start = 2.0
        chi_end = 10.0
        U_start = 8.0
        U_end = 16.0

        chi = self.tanh_chi(chi) #* 4.0  # 4.0
        chi = change_tanh_range(chi, chi_start, chi_end)
        chi0 = torch.zeros(species.shape, dtype=dtype, device=device)
        chi0[nonblank] = chi.reshape(-1)
        chi = chi0
        
        # make sure U is positive
        # U = self.sp_U(U) + self.c # softplus example
        U = self.tanh_U(U)
        U = change_tanh_range(U, U_start, U_end)
        U0 = torch.zeros(species.shape, dtype=dtype, device=device)
        U0[nonblank] = U.reshape(-1)
        U = U0

        J = coul_J_with_Hubbard_U_screening(nonblank, coordinates, U)
        # E  = 0.5 q^T * (U+J) * q + q^T * (chi)
        # dE/dq = 0 ==> (U+J)q = - chi

        # have to replace the padding diagonal 0 with 1 on A, shift A by 1.0
        A0 = torch.zeros(n_molecule, n_atom + 1, n_atom + 1, dtype=dtype, device=device)
        # put diagonal elements of padding part as 1.0
        maskd = (
            (torch.arange(n_molecule, dtype=torch.int64, device=device) * (n_atom + 1) ** 2).reshape((-1, 1))
            + (torch.arange(n_atom, dtype=torch.int64, device=device) * (n_atom + 2)).reshape((1, -1))
        ).reshape(-1)[~nonblank.reshape(-1)]
        A0.reshape(-1)[maskd] = 1.0
        # add constraint
        maskc = (
            (torch.arange(n_molecule, dtype=torch.int64, device=device) * (n_atom + 1)).reshape((-1, 1))
            + (torch.arange(n_atom, dtype=torch.int64, device=device)).reshape((1, -1))
        ).reshape(-1)[nonblank.reshape(-1)]
        A0.reshape(-1, n_atom + 1)[maskc, -1] = 1.0
        A0[:, -1, :] = A0[:, :, -1]

        # q_mol : net charge, a float number or a tensor with shape (n_molecule, )
        # chi : shape (n_molecule, n_atom)
        # phi : same as chi
        q_mol = 0.0
        b = -chi
        b0 = torch.zeros(n_molecule, n_atom + 1, dtype=dtype, device=device)
        b0[:, :-1] = b
        b0[:, -1] = q_mol

        q, Ecoul = self.exact_charge(U, chi, J, A0, b0)

        d = self.dipole(q, coordinates)

        return q.reshape(n_molecule, n_atom), Ecoul.reshape(-1, 1) * self.energy_scale, d, U, chi

    @staticmethod
    #@torch.jit.script
    def exact_charge(U, chi, J, A0, b0):
        A = torch.diag_embed(U) + J
        A0[:, :-1, :-1] += A

        q_tmp = torch.linalg.solve(A0, b0.unsqueeze(2))
        q = q_tmp[:, :-1, :]
        # q: shape (n_molecule, n_atom, 1)
        Ecoul = 0.5 * torch.matmul(q.transpose(1, 2), torch.matmul(A, q)) + torch.matmul(q.transpose(1, 2), (chi).unsqueeze(2))

        return q, Ecoul

    @staticmethod
    #@torch.jit.script
    def dipole(q, coordinates):
        """
        q : charge, shape (n_molecule, n_atom, 1)
        coordinates : shape (n_molecule, n_atom, 3)
        """
        return torch.sum(q * coordinates, dim=1)


#@torch.compile
def coul_J_with_Hubbard_U_screening(nonblank, coordinates, U):
    a_0 = 0.529177210903  # Bohr radius in Angstrom
    E_h = 27.211386245988  # Hatree energy in eV
    e2_over_four_pi_epsilon_0 = E_h * a_0

    _, n_atom, _ = coordinates.shape
    device = coordinates.device
    dtype = coordinates.dtype

    one = torch.tensor(1.0, dtype=dtype, device=device)
    zero = torch.tensor(0.0, dtype=dtype, device=device)
    # Mask for nonblank and off-diagonal elements
    mask = (nonblank.unsqueeze(1) * nonblank.unsqueeze(2)) & ~(
        torch.eye(n_atom, device=device, dtype=torch.int).to(torch.bool).unsqueeze(0)
    )
    rij = torch.where(mask, torch.linalg.norm(coordinates.unsqueeze(1) - coordinates.unsqueeze(2), dim=-1), one)
    rij_sq = torch.pow(rij, 2)
    rij_cube = torch.pow(rij, 3)
    rij_quad = torch.pow(rij, 4)
    rij_penta = torch.pow(rij, 5)

    J0 = 1.0 / rij
    TFACT = 16.0 / (5.0 * e2_over_four_pi_epsilon_0)

    # shape:(n_mol, n_atom, n_atom)
    # TI = TFACT * U
    TI = torch.where(mask, TFACT * U.unsqueeze(2).repeat(1, 1, U.shape[1]), one)
    TI2 = TI * TI
    TI3 = TI2 * TI
    TI4 = TI2 * TI2
    TI6 = TI4 * TI2

    TJ = TFACT * U.unsqueeze(1).repeat(1, U.shape[1], 1)
    TJ2 = TJ * TJ
    TJ4 = TJ2 * TJ2
    TJ6 = TJ4 * TJ2
    TI2MTJ2 = TI2 - TJ2

    same_element_mask = mask & (abs(TI - TJ) < (0.2))  # n_mol, n_atom, n_atom
    different_element_mask = mask & ~same_element_mask  # n_mol, n_atom, n_atom
    TI2MTJ2 = torch.where(different_element_mask, TI2MTJ2, one)

    EXPTI = torch.exp((-TI * rij))
    EXPTJ = torch.exp((-TJ * rij))

    # TI2MTJ2 is small, so do divsion from left to right to provide numerical stability
    SB = EXPTI * TJ4 * TI / 2.0 / TI2MTJ2 / TI2MTJ2
    SC = EXPTI * (TJ6 - 3.0 * TJ4 * TI2) / TI2MTJ2 / TI2MTJ2 / TI2MTJ2
    SE = EXPTJ * TI4 * TJ / 2.0 / TI2MTJ2 / TI2MTJ2
    SF = EXPTJ * (-(TI6 - 3.0 * TI4 * TJ2)) / TI2MTJ2 / TI2MTJ2 / TI2MTJ2

    # compile-friendly write pattern with `where` instead of in-place operation on the mask
    sub = (1.0 * (SB - (SC / rij)) + 1.0 * (SE - (SF / rij)))
    J0 = torch.where(different_element_mask,J0-sub,J0)
    
    # Taylor expansion around ta - tb -> 0
    SSB = TI3 / 48.0
    SSC = 3.0 * TI2 / 16.0
    SSD = 11.0 * TI / 16.0
    SSE = 1.0
    # first-order terms
    SSF = 5.0 / 32.0
    SSG = 5.0 * TI / 32.0
    SSH = 1.0 * TI2 / 16.0
    SSI = 1.0 * TI3 / 96.0
    # second-order terms
    SSJ = 3.0 / (32.0 * TI)
    SSK = 3.0 / 32.0
    SSL = 3.0 * TI / 64.0
    SSM = 1.0 * TI2 / 64.0
    SSN = 1.0 * TI3 / 320.0
    # third-order terms
    SSO = 3.0 / (64.0 * TI2)
    SSP = 3.0 / (64.0 * TI)
    SSQ = 5.0 / 192.0
    SSR = 1.0 * TI / 96.0
    SSS = 1.0 * TI2 / 320.0
    SST = 1.0 * TI3 / 1440.0

    # compile-friendly write pattern with `where` instead of in-place operation.
    sub = (
            EXPTI * (SSB * rij_sq + SSC * rij + SSD + SSE / rij)
            + EXPTI * (SSF + SSG * rij + SSH * rij_sq + SSI * rij_cube) * (TI - TJ)
            + EXPTI * (SSJ + SSK * rij + SSL * rij_sq + SSM * rij_cube + SSN * rij_quad) * torch.pow(TI - TJ, 2)
            + EXPTI * (SSO + SSP * rij + SSQ * rij_sq + SSR * rij_cube + SSS * rij_quad + SST * rij_penta) * torch.pow(TI - TJ, 3)
        )
    J0 = torch.where(same_element_mask, J0-sub, J0)

    J = torch.where(mask, e2_over_four_pi_epsilon_0 * J0, zero)

    return J
