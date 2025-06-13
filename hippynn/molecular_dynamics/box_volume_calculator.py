# Calculate box shape needed to achieve specified density
# Portions of this code were written with assistance from an LLM.

# Avogadro's number (molecules per mole)
AVOGADRO = 6.022e23

# Molar masses in g/mol
molar_masses = {
    "choline": 104.17,
    "chloride": 35.45,
    "urea": 60.06,
    "ethylene glycol": 62.07,
    "methanol": 32.04,
    "methane": 16.04,
    "benzene": 78.11,
}

def calculate_box_side_length(molecule_counts, density):
    """
    Calculate the side length of a cubic box that has the specified density.

    Parameters:
      molecule_counts (dict): A dictionary with keys as molecule names and values as the number of molecules.
      density (float): The desired density in g/cm³.

    Returns:
      float: side length in Angstroms.
    """
    total_mass = 0.0
    for molecule, count in molecule_counts.items():
        key = molecule.lower()
        try:
            total_mass += (count / AVOGADRO) * molar_masses[key]
        except KeyError:
            raise ValueError(
                f"Molecule '{molecule}' not found in molar_masses dictionary. Please add an entry for it and try again.\n"
                + f"Existing molecules: {", ".join(molar_masses.keys())}"
            )

    volume = total_mass / density
    side_length_cm = volume ** (1/3)
    side_length_angstrom = side_length_cm * 1e8

    return side_length_angstrom

def print_results(molecule_counts, density, side_length_ang):
    max_mol_name_length = max([len(name) for name in molecule_counts.keys()])

    print(f"Box contains:")
    for key, value in molecule_counts.items():
        print(f"  {key+":":<{max_mol_name_length+3}} {value} molecules")
    print(f"Density: {density} g/cm^3")
    print(f"Cube side length: {side_length_ang:.2f} Ang")

def main(molecule_counts, density):
    side_length_ang = calculate_box_side_length(molecule_counts, density)
    print_results(molecule_counts, density, side_length_ang)

if __name__ == "__main__":
    molecule_counts = {'choline': 250, 'chloride': 250, 'urea': 500}
    density = 1.178 # g/cm^3

    main(molecule_counts, density)

    
