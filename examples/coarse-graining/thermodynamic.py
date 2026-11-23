# This script calculates Kirkwood-Buff Integrals (KBI) from RDF data using the KBKit package (https://github.com/anl-sepsci/kbkit)
# The KBI values are further used to estimate thermodynamic properties such as compressibility, thermodynamic factors, 
# partial molar volumes, and chemical-potential derivatives. 

import csv
import re
from pathlib import Path

import numpy as np
from kbkit.kbi import KBIntegrator

# RDF data files can be found in the DES datasets: https://doi.org/10.5281/zenodo.24089819
# The datasets e.g., reline_298_COG_cg.npz, should be placed in datasets/des_cg/, where datasets/ is at the same level as the hippynn repository.
data_dir = Path("datasets/des_cg")

# deep eutectic solvent, temperature, mapping type, all-atom/coarse-grained
systems = [
    ("reline", 298, "COG", "aa"),
    # ("reline", 298, "COG", "cg"),
    # ("ethaline", 298, "COG", "aa"),
    # ("ethaline", 333, "COM", "cg"),
]

# Species IDs are choline, chloride, and the hydrogen-bond donor (urea or ethylene glycol)
species = {
    "reline": {1: "Choline", 2: "Chloride", 3: "Urea"},
    "ethaline": {1: "Choline", 2: "Chloride", 3: "Ethylene glycol"},
}

# RDF species pairs in the dataset e.g., rdf_values_1_1
pairs = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
pair_keys = tuple(f"rdf_values_{i}_{j}" for i, j in pairs)

fit_window = (1.0, 3.0)  # nm
kB = 1.380649e-23 # Boltzmann constant (J/K)
NA = 6.022141e23 # Avogadro number (mol^-1)

for liquid, temperature, mapping, model in systems:
    input_file = data_dir / f"{liquid}_{temperature}_{mapping}_{model}.npz"
    temp = temperature
    names = species[liquid]

    # KBKit works with nm by default
    with np.load(input_file) as data:
        r = np.asarray(data["rdf_bins"], dtype=float) * 0.1
        cells = np.asarray(data["cells"], dtype=float) * 0.1
        types = data["types"] if "types" in data.files else data["species"]
        labels, amounts = np.unique(np.atleast_2d(types)[0], return_counts=True)
        counts = dict(zip(labels.tolist(), amounts.tolist()))
        volume = float(np.mean(np.abs(np.linalg.det(cells))))

        mask = (r >= fit_window[0]) & (r <= fit_window[1])
        running_kbi, fits = [], []

        for (i, j), key in zip(pairs, pair_keys):
            gv = KBIntegrator(r=r, gr=data[key], n_ref=counts[j], box_volume=volume, delta=int(i == j), weight_type="geometric",).compute_running_kbi("geometric")
            running_kbi.append(gv)
            fits.append(np.polyfit(r[mask], r[mask] * gv[mask], 1))

    # Collect the six pairwise KBIs by their species IDs
    kbi = dict(zip(pairs, (fit[0] for fit in fits)))
    g = lambda i, j: kbi[i, j]
    choline, chloride, donor = 1, 2, 3
    g11 = (g(choline, choline) + 2 * g(choline, chloride) + g(chloride, chloride)) / 4
    g12 = (g(choline, donor) + g(chloride, donor)) / 2
    g22 = g(donor, donor)
    rho_ions = (counts[choline] + counts[chloride]) / volume
    rho_donor = counts[donor] / volume
    d = rho_ions + rho_donor + rho_ions * rho_donor * (g11 + g22 - 2 * g12)

    # Compressibility, thermodynamic factor, and partial molar volumes
    numerator = 1 + rho_ions * g11 + rho_donor * g22
    numerator += rho_ions * rho_donor * (g11 * g22 - g12 ** 2)
    kappa = numerator / (kB * temp * d) * 1e-18
    gamma = (rho_ions + rho_donor) / d
    v_ions = (1 + rho_donor * (g22 - g12)) / d * NA * 1e-21
    v_donor = (1 + rho_ions * (g11 - g12)) / d * NA * 1e-21
    rgas = NA * kB
    mu_ions = rgas * temp / (rho_ions * (1 + rho_ions * (g11 - g12)))
    mu_donor = rgas * temp / (rho_donor * (1 + rho_donor * (g22 - g12)))

    kbi_file = input_file.with_name(f"{input_file.stem}_kbi.npz")
    np.savez_compressed(
        kbi_file,
        r_nm=r,
        pair_keys=pair_keys,
        pair_names=[f"{names[i]}–{names[j]}" for i, j in pairs],
        Gv_nm3=running_kbi,
        G_infinity_nm3=[fit[0] for fit in fits],
        intercept_nm4=[fit[1] for fit in fits],
        species_ids=sorted(counts),
        species_names=[names[i] for i in sorted(counts)],
        species_counts=[counts[i] for i in sorted(counts)],
        mean_volume_nm3=volume,
        temperature_k=temp,
    )
    
    results = {
        "g_ions_ions_nm3": g11, "g_ions_donor_nm3": g12, "g_donor_donor_nm3": g22,
        "kappa_gpa-1": kappa, "gamma": gamma,
        "v_grouped_ions_cm3_mol": v_ions, "v_donor_cm3_mol": v_donor,
        "mu_derivative_ions": mu_ions, "mu_derivative_donor": mu_donor,
    }
    thermo_file = input_file.with_name(f"{input_file.stem}_thermodynamics.csv")
    with thermo_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results)
        writer.writeheader()
        writer.writerow(results)

    print(f"Saved {kbi_file}")
    print(f"Saved {thermo_file}")
