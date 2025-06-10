import numpy as np

class SpeciesLookup:
    def __init__(self, species_numbers, species_names):
        pairs = np.stack((species_numbers.reshape(-1), species_names.reshape(-1)), axis=1)
        unique_pairs = np.unique(pairs, axis=0)
        if len(unique_pairs) > len(np.unique(unique_pairs[:, 0])) or len(unique_pairs) > len(np.unique(unique_pairs[:, 1])):
            raise ValueError("Species number to name pairing is not unique!")
        
        self._number_to_name = dict(pairs)
        self._name_to_number = {name: num for num, name in pairs}

    def number_to_name(self, num):
        return self._number_to_name[str(num)]
    
    def name_to_number(self, name):
        return self._name_to_number[name]