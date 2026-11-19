# This script calculates Kirkwood-Buff Integrals (KBI) from RDF data using the KBKit package (https://github.com/anl-sepsci/kbkit)
# The KBI values are further used to estimate thermodynamic properties such as compressibility, thermodynamic factors, 
# partial molar volumes, and chemical-potential derivatives. It also compares CG and Martini
# results against the corresponding AA reference.

# RDF data files can be found in the DES datasets: https://doi.org/10.5281/zenodo.24089819
# The datasets e.g., reline_298_COG_cg.npz, should be placed in the same directory as the script.

import csv
import json
import re
from pathlib import Path

import numpy as np
from kbkit.kbi import KBIntegrator


# RDF bins and cell vectors in the input files are in angstroms.
ANGSTROM_TO_NM = 0.1
PAIR_KEYS = (
    "rdf_values_1_1", "rdf_values_1_2", "rdf_values_1_3",
    "rdf_values_2_2", "rdf_values_2_3", "rdf_values_3_3",
)
SYSTEMS = (
    ("ethaline", 298, "COG"), ("ethaline", 333, "COG"),
    ("ethaline", 298, "COM"), ("ethaline", 333, "COM"),
    ("reline", 298, "COG"), ("reline", 333, "COG"),
    ("reline", 298, "COM"), ("reline", 333, "COM"),
    ("reline", 298, "COG_10x"),
)
SPECIES = {
    "ethaline": {1: "Choline", 2: "Chloride", 3: "Ethylene glycol"},
    "reline": {1: "Choline", 2: "Chloride", 3: "Urea"},
}

# The regression is performed on L G^V(L), in nm.
FIT_MIN, FIT_MAX = 1.0, 3.0
MIN_WIDTH, MIN_POINTS = 1.0, 10
FIXED_WINDOWS = {
    # ("reline", 333, "COM", "rdf_values_2_3"): (1.0, 3.0),
}

kB = 1.380649e-23 # Boltzmann's constant in J/K. Used for thermo calculations
NA = 6.022141e23 # Avogadro's number. Used for thermo calculations

# Temperature in K. All parameters are listed in the Supplemental Material
temperatures = {298: 298.15, 333: 333.15}
property_names = (
    "kappa (GPa^-1)", "Gamma", "V_grouped_ions (cm^3/mol)",
    "V_donor (cm^3/mol)", "mu derivative 1", "mu derivative 2",
)


def clean_name(name):
    name = re.sub(r"[\s_-]+", "", str(name).lower())
    aliases = {
        "ch": "choline", "cl": "chloride", "u": "urea", "ur": "urea",
        "eg": "ethyleneglycol",
    }
    return aliases.get(name, name)


def pair_ids(key):
    match = re.fullmatch(r"rdf_values_(\d+)_(\d+)", key)
    return tuple(map(int, match.groups()))


def fit_line(x, y):
    if len(x) < MIN_POINTS or x[-1] - x[0] < MIN_WIDTH:
        return None
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None

    dx, dy = x - x.mean(), y - y.mean()
    xx, yy = np.dot(dx, dx), np.dot(dy, dy)
    if xx == 0 or yy == 0:
        return None

    slope = float(np.dot(dx, dy) / xx)
    intercept = float(y.mean() - slope * x.mean())
    residual = y - (slope * x + intercept)
    r2 = float(1.0 - np.dot(residual, residual) / yy)
    return slope, intercept, r2


def find_window(r, gv, bounds=None):
    lo, hi = (FIT_MIN, FIT_MAX) if bounds is None else bounds
    if bounds is not None and (lo < r[0] or hi > r[-1]):
        return None
    valid = np.flatnonzero((r >= lo) & (r <= hi) & np.isfinite(gv))
    chunks = np.split(valid, np.flatnonzero(np.diff(valid) != 1) + 1)
    best = None

    for chunk in chunks:
        for first in range(len(chunk) - MIN_POINTS + 1):
            for last in range(first + MIN_POINTS, len(chunk) + 1):
                idx = chunk[first:last]
                fit = fit_line(r[idx], r[idx] * gv[idx])
                if fit is None:
                    continue
                candidate = (*fit, float(r[idx[0]]), float(r[idx[-1]]))
                if best is None or (candidate[2], candidate[4] - candidate[3]) > (best[2], best[4] - best[3]):
                    best = candidate
    return best


def locate(root, name):
    paths = list(root.rglob(name))
    if not paths:
        raise FileNotFoundError(f"Could not find {name} below {root}")
    if len(paths) > 1:
        raise RuntimeError(f"More than one copy of {name} was found:\n" + "\n".join(map(str, paths)))
    return paths[0]


def files_for(root, liquid, temperature, mapping):
    files = {
        "AA": locate(root, f"{liquid}_{temperature}_{mapping}_aa.npz"),
        "CG": locate(root, f"{liquid}_{temperature}_{mapping}_cg.npz"),
    }

    if liquid == "reline" and mapping == "COM":
        files["Martini"] = locate(root, f"reline_{temperature}_Martini_cg.npz")
    return files


def read_file(path, liquid):
    with np.load(path) as data:
        if "type_to_mol_name" in data.files:
            names = data["type_to_mol_name"].item()
            if isinstance(names, bytes):
                names = names.decode()
            if isinstance(names, str):
                names = json.loads(names)
            names = {int(i): name for i, name in names.items()}
            for i, expected in SPECIES[liquid].items():
                if clean_name(names.get(i, "")) != clean_name(expected):
                    raise ValueError(f"{path}: species {i} does not match the {liquid} mapping")

        r = np.asarray(data["rdf_bins"], dtype=float) * ANGSTROM_TO_NM
        cells = np.asarray(data["cells"], dtype=float) * ANGSTROM_TO_NM
        if "types" in data.files:
            types = np.asarray(data["types"])
        elif "species" in data.files:
            types = np.asarray(data["species"])
        else:
            raise KeyError(f"{path}: neither 'types' nor 'species' was found")
        if (r.ndim != 1 or len(r) < MIN_POINTS or not np.all(np.isfinite(r))
                or np.any(r < 0) or np.any(np.diff(r) <= 0)):
            raise ValueError(f"{path}: invalid RDF bins")
        if cells.ndim != 3 or cells.shape[1:] != (3, 3):
            raise ValueError(f"{path}: expected cells with shape (frames, 3, 3)")
        if types.ndim != 2 or types.shape[1] == 0 or types.shape[0] not in (1, len(cells)):
            raise ValueError(f"{path}: invalid types array")

        volume = np.abs(np.linalg.det(cells))
        if np.any(~np.isfinite(volume)) or np.any(volume <= 0):
            raise ValueError(f"{path}: invalid cell volume")
        labels, amounts = np.unique(types[0], return_counts=True)
        counts = dict(zip(labels.tolist(), amounts.tolist()))
        if len(types) > 1:
            for label, count in counts.items():
                if not np.all(np.count_nonzero(types == label, axis=1) == count):
                    raise ValueError(f"{path}: species counts change between frames")

        rdfs = {}
        for key in PAIR_KEYS:
            if key in data.files:
                gr = np.asarray(data[key], dtype=float)
                if gr.shape != r.shape or not np.all(np.isfinite(gr)):
                    raise ValueError(f"{path}: invalid {key}")
                rdfs[key] = gr
        if not rdfs:
            raise ValueError(f"{path}: no RDFs with the expected names")
    return r, float(volume.mean()), counts, rdfs


def save_results(path, r, values, liquid, counts, volume, temperature):
    pairs = [key for key in PAIR_KEYS if key in values]
    fits = [values[key][1] for key in pairs]
    gv = np.stack([values[key][0] for key in pairs])
    ids = np.asarray(sorted(counts), dtype=int)
    np.savez_compressed(
        path,
        r_nm=r,
        pair_keys=np.asarray(pairs),
        pair_names=np.asarray([
            f"{SPECIES[liquid][pair_ids(key)[0]]}–{SPECIES[liquid][pair_ids(key)[1]]}"
            for key in pairs
        ]),
        Gv_nm3=gv,
        LGv_nm4=gv * r,
        G_infinity_nm3=np.asarray([fit[0] if fit else np.nan for fit in fits]),
        intercept_nm4=np.asarray([fit[1] if fit else np.nan for fit in fits]),
        r2=np.asarray([fit[2] if fit else np.nan for fit in fits]),
        species_ids=ids,
        species_names=np.asarray([SPECIES[liquid][i] for i in ids]),
        species_counts=np.asarray([counts[i] for i in ids]),
        mean_volume_nm3=np.asarray(volume),
        temperature_K=np.asarray(temperature),
    )


# Print a quick guide to the contents of each output file.
def describe_output():
    print("\nEach KBI npz file contains:")
    print("  r_nm                (n_bins,)          RDF distance grid in nm")
    print("  pair_keys           (n_pairs,)         RDF key for each species pair")
    print("  pair_names          (n_pairs,)         readable names for those pairs")
    print("  Gv_nm3              (n_pairs, n_bins)  running KBIs in nm^3")
    print("  LGv_nm4             (n_pairs, n_bins)  L times the running KBI")
    print("  G_infinity_nm3      (n_pairs,)         fitted long-range KBI")
    print("  intercept_nm4       (n_pairs,)         intercept from that fit")
    print("  r2                  (n_pairs,)         coefficient of determination R^2")
    print("  species_ids         (n_species,)       type IDs from the source file")
    print("  species_names       (n_species,)       names matching those IDs")
    print("  species_counts      (n_species,)       molecule counts from the types array")
    print("  mean_volume_nm3     ()                 average box volume")
    print("  temperature_K       ()                 simulation temperature")

def thermo_from_kbi(path):
    with np.load(path) as data:
        pairs = {
            tuple(sorted(pair.split("–"))): value
            for pair, value in zip(data["pair_names"], data["G_infinity_nm3"])
        }
        counts = {
            str(name): int(count)
            for name, count in zip(data["species_names"], data["species_counts"])
        }
        volume = float(data["mean_volume_nm3"])
        T = float(data["temperature_K"])

    def G(a, b):
        try:
            return pairs[tuple(sorted((a, b)))]
        except KeyError as error:
            raise ValueError(f"{path}: missing KBI for {a}–{b}") from error

    donor = [name for name in counts if name not in ("Choline", "Chloride")]
    if len(donor) != 1:
        raise ValueError(f"{path}: could not identify the donor species")
    donor = donor[0]

    G11 = (G("Choline", "Choline") + 2 * G("Choline", "Chloride")
           + G("Chloride", "Chloride")) / 4
    G12 = (G("Choline", donor) + G("Chloride", donor)) / 2
    G22 = G(donor, donor)
    G_grouped_ions = np.array([[G11, G12], [G12, G22]])

    rho_ions = (counts["Choline"] + counts["Chloride"]) / volume
    rho_donor = counts[donor] / volume
    D = rho_ions + rho_donor + rho_ions * rho_donor * (G11 + G22 - 2 * G12)

    numerator = 1 + rho_ions * G11 + rho_donor * G22
    numerator += rho_ions * rho_donor * (G11 * G22 - G12 ** 2)
    kappa = numerator / (kB * T * D) * 1e-18
    Gamma = (rho_ions + rho_donor) / D
    V_ion = (1 + rho_donor * (G22 - G12)) / D * NA * 1e-21
    V_donor = (1 + rho_ions * (G11 - G12)) / D * NA * 1e-21

    R = NA * kB
    D1 = 1 + rho_ions * (G11 - G12)
    D2 = 1 + rho_donor * (G22 - G12)
    mu_ion = R * T / (rho_ions * D1)
    mu_donor = R * T / (rho_donor * D2)

    properties = np.array([kappa, Gamma, V_ion, V_donor, mu_ion, mu_donor])
    return G_grouped_ions, properties


def write_thermo(path, KB_results, thermo_results):
    columns = (
        "kappa_GPa-1", "Gamma", "V_grouped_ions_cm3_mol",
        "V_donor_cm3_mol", "mu_derivative_ions", "mu_derivative_donor",
    )
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "system", "model", "G_ions_ions_nm3", "G_ions_donor_nm3",
            "G_donor_donor_nm3", *columns,
        ))
        writer.writeheader()
        for system, models in KB_results.items():
            for model, G in models.items():
                properties = thermo_results[system][model]
                row = {
                    "system": system,
                    "model": model,
                    "G_ions_ions_nm3": G[0, 0],
                    "G_ions_donor_nm3": G[0, 1],
                    "G_donor_donor_nm3": G[1, 1],
                }
                row.update(dict(zip(columns, properties)))
                writer.writerow(row)


def write_errors(path, KB_results, thermo_results):
    error_sum = np.zeros(len(property_names))
    comparisons = 0

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "system", "AA_reference", "quantity", "AA_value", "CG_value", "percent_error",
        ))
        writer.writeheader()

        for system, models in KB_results.items():
            if "cg" not in models:
                continue
            reference = system.replace("_Martini", "_COM")
            if reference not in KB_results or "aa" not in KB_results[reference]:
                raise ValueError(f"No AA reference was found for {system}")

            G_aa = KB_results[reference]["aa"]
            G_cg = models["cg"]
            properties_aa = thermo_results[reference]["aa"]
            properties_cg = thermo_results[system]["cg"]
            with np.errstate(divide="ignore", invalid="ignore"):
                G_error = 100 * (G_cg - G_aa) / G_aa
                property_error = 100 * (properties_cg - properties_aa) / properties_aa

            values = (
                ("G_ions_ions_nm3", G_aa[0, 0], G_cg[0, 0], G_error[0, 0]),
                ("G_ions_donor_nm3", G_aa[0, 1], G_cg[0, 1], G_error[0, 1]),
                ("G_donor_donor_nm3", G_aa[1, 1], G_cg[1, 1], G_error[1, 1]),
                *zip(property_names, properties_aa, properties_cg, property_error),
            )
            for name, aa, cg, error in values:
                writer.writerow({
                    "system": system,
                    "AA_reference": reference,
                    "quantity": name,
                    "AA_value": aa,
                    "CG_value": cg,
                    "percent_error": error,
                })

            error_sum += property_error
            comparisons += 1

    if comparisons:
        return error_sum / comparisons
    return None


def main():
    root = Path.cwd()
    output = root / "kbi_results"
    output.mkdir(parents=True, exist_ok=True)
    KB_files = []

    for liquid, temperature, mapping in SYSTEMS:
        for source in files_for(root, liquid, temperature, mapping).values():

            r, volume, counts, rdfs = read_file(source, liquid)
            values = {}
            for key, gr in rdfs.items():
                i, j = pair_ids(key)
                if i not in counts or j not in counts:
                    raise ValueError(f"{source}: {key} refers to a missing species")
                gv = np.asarray(KBIntegrator(
                    r=r, gr=gr, n_ref=counts[j], box_volume=volume,
                    delta=int(i == j), weight_type="geometric",
                ).compute_running_kbi("geometric"), dtype=float)
                if gv.shape != r.shape:
                    raise ValueError(f"{source}: unexpected KBI shape for {key}")
                fit = find_window(r, gv, FIXED_WINDOWS.get((liquid, temperature, mapping, key)))
                values[key] = (gv, fit)
                if fit is None:
                    print(f"{source.name:40} {key}: no fit")
                else:
                    print(f"{source.name:40} {key}: G∞ = {fit[0]:.6g}, R² = {fit[2]:.5f}")

            target = output / f"{source.stem}_kbi.npz"
            save_results(target, r, values, liquid, counts, volume, temperatures[temperature])
            KB_files.append(target)
            print(f"Saved {target.relative_to(root)}")

    KB_results = {}
    thermo_results = {}
    for path in KB_files:
        G, properties = thermo_from_kbi(path)
        system, model = path.stem.replace("_kbi", "").rsplit("_", 1)
        KB_results.setdefault(system, {})[model] = G
        thermo_results.setdefault(system, {})[model] = properties

    thermo_file = output / "thermodynamic_properties.csv"
    error_file = output / "thermodynamic_errors.csv"
    write_thermo(thermo_file, KB_results, thermo_results)
    average_error = write_errors(error_file, KB_results, thermo_results)
    print(f"Saved {thermo_file.relative_to(root)}")
    print(f"Saved {error_file.relative_to(root)}")

    if average_error is not None:
        print("\nAverage percent errors:")
        for name, error in zip(property_names, average_error):
            print(f"{name}: {error:.3f}%")
        print(f"Average partial molar volume error: {(average_error[2] + average_error[3]) / 2:.3f}%")
        print(f"Average chemical-potential derivative error: {(average_error[4] + average_error[5]) / 2:.3f}%")

    describe_output()


if __name__ == "__main__":
    main()
