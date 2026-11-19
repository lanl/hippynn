# This script calculates Kirkwood-Buff Integrals (KBI) from RDF data using the KBKit package (https://github.com/anl-sepsci/kbkit)
# The KBI values are further used to estimate thermodynamic properties such as compressibility, thermodynamic factors, 
# partial molar volumes, and chemical-potential derivatives. It also compares CG and Martini
# results against the corresponding AA reference.

# RDF data files can be found in the DES datasets: https://doi.org/10.5281/zenodo.24089819
# The datasets e.g., reline_298_COG_cg.npz, should be placed in the same directory as the script.

import csv
import re
from pathlib import Path

import numpy as np
from kbkit.kbi import KBIntegrator


INPUT_FILES = [
    "reline_298_COG_aa.npz",
    # "reline_298_COG_cg.npz",
    # "ethaline_298_COG_aa.npz",
    # "ethaline_333_COM_cg.npz",
]

SPECIES = {
    "reline": {1: "Choline", 2: "Chloride", 3: "Urea"},
    "ethaline": {1: "Choline", 2: "Chloride", 3: "Ethylene glycol"},
}

FIT_WINDOW = (1.0, 3.0)  # nm
kB = 1.380649e-23
NA = 6.022141e23
PAIR_KEYS = (
    "rdf_values_1_1", "rdf_values_1_2", "rdf_values_1_3",
    "rdf_values_2_2", "rdf_values_2_3", "rdf_values_3_3",
)


def pair_ids(key):
    return tuple(map(int, re.findall(r"\d+", key)))


def run(input_file):
    liquid = next(name for name in SPECIES if input_file.name.startswith(f"{name}_"))
    match = re.search(r"_(298|333)_", input_file.name)
    if match is None:
        raise ValueError(f"Could not infer temperature from {input_file.name}")
    temperature = float(match.group(1)) + 0.15
    names = SPECIES[liquid]

    with np.load(input_file) as data:
        r = np.asarray(data["rdf_bins"], dtype=float) * 0.1
        cells = np.asarray(data["cells"], dtype=float) * 0.1
        types = data["types"] if "types" in data.files else data["species"]
        types = np.atleast_2d(types)
        labels, amounts = np.unique(types[0], return_counts=True)
        counts = dict(zip(labels.tolist(), amounts.tolist()))
        volume = float(np.mean(np.abs(np.linalg.det(cells))))
        rdfs = {key: np.asarray(data[key], dtype=float) for key in PAIR_KEYS}

    mask = (r >= FIT_WINDOW[0]) & (r <= FIT_WINDOW[1])
    if mask.sum() < 3:
        raise ValueError("The fitting window contains fewer than three RDF bins.")

    running_kbi, fits = [], []
    for key, gr in rdfs.items():
        i, j = pair_ids(key)
        Gv = KBIntegrator(
            r=r, gr=gr, n_ref=counts[j], box_volume=volume,
            delta=int(i == j), weight_type="geometric",
        ).compute_running_kbi("geometric")
        running_kbi.append(Gv)
        fits.append(np.polyfit(r[mask], r[mask] * Gv[mask], 1))

    kbi = {tuple(sorted(pair_ids(key))): fit[0] for key, fit in zip(PAIR_KEYS, fits)}
    G = lambda i, j: kbi[tuple(sorted((i, j)))]
    choline = next(i for i, name in names.items() if name.lower() == "choline")
    chloride = next(i for i, name in names.items() if name.lower() == "chloride")
    donor = next(i for i in names if i not in (choline, chloride))

    # Combine choline and chloride into one ionic component.
    G11 = (G(choline, choline) + 2 * G(choline, chloride) + G(chloride, chloride)) / 4
    G12 = (G(choline, donor) + G(chloride, donor)) / 2
    G22 = G(donor, donor)
    rho_ions = (counts[choline] + counts[chloride]) / volume
    rho_donor = counts[donor] / volume
    D = rho_ions + rho_donor + rho_ions * rho_donor * (G11 + G22 - 2 * G12)

    numerator = 1 + rho_ions * G11 + rho_donor * G22
    numerator += rho_ions * rho_donor * (G11 * G22 - G12 ** 2)
    kappa = numerator / (kB * temperature * D) * 1e-18
    Gamma = (rho_ions + rho_donor) / D
    V_ions = (1 + rho_donor * (G22 - G12)) / D * NA * 1e-21
    V_donor = (1 + rho_ions * (G11 - G12)) / D * NA * 1e-21
    R = NA * kB
    mu_ions = R * temperature / (rho_ions * (1 + rho_ions * (G11 - G12)))
    mu_donor = R * temperature / (rho_donor * (1 + rho_donor * (G22 - G12)))

    species_ids = np.asarray(sorted(counts))
    pair_names = [f"{names[i]}–{names[j]}" for i, j in map(pair_ids, PAIR_KEYS)]
    kbi_file = input_file.with_name(f"{input_file.stem}_kbi.npz")
    np.savez_compressed(
        kbi_file,
        r_nm=r,
        pair_keys=np.asarray(PAIR_KEYS),
        pair_names=np.asarray(pair_names),
        Gv_nm3=np.asarray(running_kbi),
        G_infinity_nm3=np.asarray([fit[0] for fit in fits]),
        intercept_nm4=np.asarray([fit[1] for fit in fits]),
        species_ids=species_ids,
        species_names=np.asarray([names[i] for i in species_ids]),
        species_counts=np.asarray([counts[i] for i in species_ids]),
        mean_volume_nm3=volume,
        temperature_K=temperature,
    )

    results = {
        "G_ions_ions_nm3": G11,
        "G_ions_donor_nm3": G12,
        "G_donor_donor_nm3": G22,
        "kappa_GPa-1": kappa,
        "Gamma": Gamma,
        "V_grouped_ions_cm3_mol": V_ions,
        "V_donor_cm3_mol": V_donor,
        "mu_derivative_ions": mu_ions,
        "mu_derivative_donor": mu_donor,
    }
    thermo_file = input_file.with_name(f"{input_file.stem}_thermodynamics.csv")
    with thermo_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results)
        writer.writeheader()
        writer.writerow(results)

    print(f"Saved {kbi_file}")
    print(f"Saved {thermo_file}")


for filename in INPUT_FILES:
    run(Path(filename))



if __name__ == "__main__":
    main()
