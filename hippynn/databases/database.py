"""
Base database functionality from dictionary of numpy arrays
"""

from typing import Union
import warnings
import numpy as np
import torch
from pathlib import Path

from .restarter import NoRestart
from ..tools import arrdict_len, device_fallback, unsqueeze_multiple

from torch.utils.data import DataLoader, TensorDataset, Subset

from collections import defaultdict

_AUTO_SPLIT_PREFIX = "split_mask_"

class Database:
    """
    Class for holding a pytorch dataset, splitting it, generating dataloaders, etc."
    """

    def __init__(
        self,
        arr_dict: dict[str, torch.Tensor],
        inputs: list[str],
        targets: list[str],
        seed: [int, torch.Generator],
        test_size: Union[float, int] = None,
        valid_size: Union[float, int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        allow_unfound: bool = False,
        auto_split: bool = False,
        device: torch.device = None,
        dataloader_kwargs: dict[str, object] = None,
        quiet=False,
    ):
        """
        :param arr_dict: dictionary mapping strings to numpy arrays
        :param inputs:   list of strings for input db_names
        :param targets:  list of strings for output db_namees
        :param seed:     int, for random splitting, or existing torch.Generator object
        :param test_size: fraction of data to use in test split
        :param valid_size: fraction of data to use in train split
        :param num_workers: passed to pytorch dataloaders
        :param pin_memory: passed to pytorch dataloaders
        :param allow_unfound: If true, skip checking if the needed inputs and targets are found.
           This allows setting inputs=None and/or targets=None.
        :param auto_split: If true, look for keys like "split_*" to make initial splits from. See write_npz() method.
        :param device: if set, move the dataset to this device after splitting.
        :param dataloader_kwargs: dictionary, passed to pytorch dataloaders in addition to num_workers, pin_memory.
           Refer to pytorch documentation for details.
        :param quiet: If True, print little or nothing while loading.
        """

        # Restartable Children of this class should change this after calling super().__init__() .
        self.restarter = NoRestart()

        self.inputs = inputs
        self.targets = targets
        self.quiet = quiet
        self.splitting_completed = False
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.auto_split = auto_split

        self.arr_dict = {}
        for k, v in arr_dict.items():
            try:
                self.arr_dict[k] = torch.as_tensor(v)
            except Exception as e:
                warnings.warn(f"Skipping key '{k}': could not convert to tensor ({type(v)}), reason: {e}")

        if not quiet:
            print(f"All arrays:")
            prettyprint_arrays(self.arr_dict)

        try:
            _var_list = self.var_list
        except RuntimeError:
            if not quiet:
                print(
                    "Database inputs and/or targets not specified. "
                    "The database will not be checked against and model inputs and targets (db_info)."
                )
            _var_list = []

        for k in _var_list:
            if k not in self.arr_dict and k not in ("indices", "split_indices"):
                if allow_unfound:
                    warnings.warn(f"Required database quantity '{k}' not present during database initialization.")
                else:
                    raise KeyError(
                        f"Array dictionary missing required variable:'{k}'."
                        "Pass allow_unfound=True to avoid checking of inputs targets."
                    )
        if not quiet and _var_list and not allow_unfound:
            print("Finished checking input and target arrays; all necessary arrays were found.")

        if "indices" not in self.arr_dict:
            if not quiet:
                print("Database: Using auto-generated data indices")
            self.arr_dict["indices"] = torch.arange(len(self), dtype=int)
        else:
            if not quiet:
                print("Database: Using pre-specified data indices.")

        self.splits = {}

        if isinstance(seed, torch.Generator):
            self.random_state = seed
        else:
            self.random_state = torch.Generator()
            self.random_state.manual_seed(seed)

        if self.auto_split:
            if test_size is not None or valid_size is not None:
                warnings.warn(
                    f"Auto split was set but test and valid size was also set."
                    f" Ignoring supplied test and validation sizes ({test_size} and {valid_size}."
                )
            self.make_automatic_splits()

        if test_size is not None or valid_size is not None:
            if test_size is None or valid_size is None:
                raise ValueError("Both test_size and valid_size must be set for splitting when creating a database.")
            else:
                self.make_trainvalidtest_split(test_size=test_size, valid_size=valid_size)

        if device is not None:
            if not self.splitting_completed:
                raise ValueError("Device cannot be set in constructor unless automatic split provided.")
            else:
                self.send_to_device(device)

        self.dataloader_kwargs = dataloader_kwargs.copy() if dataloader_kwargs else {}

    def __len__(self):
        return arrdict_len(self.arr_dict)

    @property
    def var_list(self):
        if self.inputs is None:
            raise RuntimeError(f"Database inputs not defined, set {Database}.inputs.")
        if self.targets is None:
            raise RuntimeError(f"Database inputs not defined, set {Database}.targets.")
        return self.inputs + self.targets

    def send_to_device(self, device: torch.device = None):
        """
        Move the database to an accelerator device if possible.
        In some circumstances this can accelerate training.

        .. Note::
           If the database is moved to a GPU,
           pin_memory will be set to False
           and num_workers will be set to 0.

        :param device: device to move to, if None, try to auto-detect.
        :return:
        """
        if len(self.splits) == 0:
            raise RuntimeError("Arrays must be split before sending database to device.")
        if device is None:
            device = device_fallback()
        else:
            device = torch.device(device)
        if device.type != "cpu":
            if self.pin_memory:
                warnings.warn("Pin memory was set, but target device requested is not CPU. Setting pin_memory=False.")
                self.pin_memory = False
            if self.num_workers != 0:
                warnings.warn("Num workers was set, but target device requested is not CPU. Setting num_workers=0.")
                self.num_workers = 0

        for split, arrdict in self.splits.items():
            for k in arrdict:
                arrdict[k] = arrdict[k].to(device)
        return

    def make_random_split(self, split_name: str, split_size: Union[int, float]):
        """
        Make a random split using self.random_state to select items.

        :param split_name: String naming the split, can be anything, but 'train', 'valid', and 'test' are special.
        :param split_size: int (number of items) or float<1, fraction of samples.
        :return:
        """
        if self.splitting_completed:
            raise RuntimeError("Database already split!")

        if split_size < 1:
            split_size = int(split_size * len(self))

        perm = torch.randperm(len(self.arr_dict["indices"]), generator=self.random_state)
        split_indices = self.arr_dict["indices"][perm[:split_size]]

        split_indices.sort()

        return self.make_explicit_split(split_name, split_indices)

    def make_trainvalidtest_split(self, *, test_size: Union[int, float], valid_size: Union[int, float]):
        """
        Make a split for train, valid, and test out of any remaining unsplit entries in the database.
        The size is specified in terms of test and valid splits; the train split will be the remainder.

        If you wish to specify precise rows for each split, see `make_explict_split`
        or `make_explicit_split_bool`.

        This function takes keyword-arguments only in order to prevent confusion over which
        size is which.

        The types of both test_size and valid_size parameters must match.

        :param test_size: int (count) or float (fraction) of data to assign to test split
        :param valid_size: int (count) or float (fraction) of data to assign to valid split
        :return: None
        """
        if self.splitting_completed:
            raise RuntimeError("Database already split!")

        if valid_size < 1:
            if test_size >= 1:
                raise ValueError("If train or valid size is set as a fraction, then set test_size as a fraction")
            else:
                if valid_size + test_size > 1:
                    raise ValueError(f"Test fraction ({test_size}) plus valid fraction " f"({valid_size}) are greater than 1!")
                valid_size /= 1 - test_size

        self.make_random_split("test", test_size)
        self.make_random_split("valid", valid_size)
        self.split_the_rest("train")
        return

    def make_explicit_split(self, split_name:str, split_indices: torch.Tensor):
        """

        :param split_name: name for split, typically 'train', 'valid', 'test'
        :param split_indices: the indices of the items for the split
        :return:
        """
        if self.splitting_completed:
            raise RuntimeError("Database splitting already complete!")

        if len(split_indices) == 0:
            raise ValueError("Cannot make split of size 0.")
        # Compute which indices are not being split off.
        index_mask = compute_index_mask(split_indices, self.arr_dict["indices"])
        complement_mask = ~index_mask

        # Precompute the actual integer indices, because indexing with a boolean mask
        # requires doing this, and we have to index with a boolean several times.
        where_index = torch.where(index_mask)
        where_complement = torch.where(complement_mask)

        # Split off data, and keep the rest.
        self.splits[split_name] = {k: self.arr_dict[k][where_index] for k in self.arr_dict}
        if "split_indices" not in self.splits[split_name]:
            if not self.quiet:
                print(f"Adding split indices for split: {split_name}")
            self.splits[split_name]["split_indices"] = torch.arange(len(split_indices), dtype=torch.int64)

        for k, v in self.arr_dict.items():
            self.arr_dict[k] = v[where_complement]

        if not self.quiet:
            print(f"Arrays for split: {split_name}")
            prettyprint_arrays(self.splits[split_name])

        if arrdict_len(self.arr_dict) == 0:
            if not self.quiet:
                print("Database: Splitting complete.")
            self.splitting_completed = True
        return

    def make_explicit_split_bool(self, split_name: str,
                                 split_mask: Union[np.ndarray, torch.tensor]):
        """

        :param split_name: name for split, typically 'train', 'valid', 'test'
        :param split_mask: a boolean array for where to split
        :return:
        """
        if isinstance(split_mask, np.ndarray):
            split_mask = torch.as_tensor(split_mask)
        if split_mask.dtype != torch.bool:
            is_valid = ((split_mask == 0) | (split_mask == 1)).all().item()
            if not is_valid:
                raise ValueError(f"Mask function contains invalid values. Values found: {torch.unique(split_mask).tolist()}")
            else:
                split_mask = split_mask.to(torch.bool)

        indices = self.arr_dict["indices"][split_mask]
        self.make_explicit_split(split_name, indices)
        return

    def split_the_rest(self, split_name: str):
        self.make_explicit_split(split_name, self.arr_dict["indices"])
        self.splitting_completed = True
        return

    def add_split_masks(self, dict_to_add_to=None, split_prefix=None):
        """
        Add split masks to the dataset. This function is used internally before writing databases.

        This function writes tensors.
        :param dict_to_add_to: where to put the split masks. Default to self.splits.
        :param split_prefix: prefix for mask names
        :return:
        """

        if not self.splitting_completed:
            raise ValueError("Can't add split masks until splitting is complete.")

        if split_prefix is None:
            split_prefix = _AUTO_SPLIT_PREFIX

        if dict_to_add_to is None:
            dict_to_add_to = self.splits

        for s in self.splits.keys():
            mask_name = split_prefix + s
            for sprime, split in self.splits.items():

                if sprime == s:
                    mask = torch.ones_like(split["indices"], dtype=torch.bool)
                else:
                    mask = torch.zeros_like(split["indices"], dtype=torch.bool)

                if mask_name in split:
                    # Check that the mask is correct and in the dict
                    old_mask = dict_to_add_to[sprime][mask_name]
                    if (old_mask != mask).all():
                        raise ValueError(f"Mask in database did not match existing split structure: {mask_name} ")
                else:
                    # if not present, write it.
                    dict_to_add_to[sprime][mask_name] = mask

    def make_automatic_splits(self, split_prefix=None, dry_run=False):
        """
        Split the database automatically. Since the user specifies this routine,
        it fails pretty strictly.

        :param split_prefix: None, use default.
          If otherwise, use this prefix to determine what arrays are masks.
        :param dry_run: Only validate that existing split masks are correct; don't perform splitting.
        :return:
        """

        if split_prefix is None:
            split_prefix = _AUTO_SPLIT_PREFIX
        if not self.quiet:
            print("Attempting automatically splitting.")
        # Find mask-like variables
        mask_vars = set()

        # Here we validate existing masks.
        # We want to make sure that if someone did it manually there was not a mistake.
        for k, arr in self.arr_dict.items():
            if k.startswith(split_prefix):
                if arr.ndim != 1:
                    raise ValueError(f"Split mask for '{k}' has too many dimensions. Shape: {arr.shape=}")
                if arr.dtype == torch.bool:
                    mask_vars.add(k)
                elif arr.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                    if ((arr == 0) | (arr == 1)).all():
                        mask_vars.add(k)
                    else:
                        arr_values = torch.unique(arr).tolist()
                        raise ValueError(f"Integer masks for split contain invalid values: {arr_values}")
                else:
                    raise ValueError(f"Failed on split {k} Split arrays must be 1-d boolean or (0,1)-valued integer arrays.")


        if not len(mask_vars):
            raise ValueError("No split mask detected.")

        masks = {k[len(split_prefix) :]: self.arr_dict[k].to(torch.bool) for k in mask_vars}

        if not self.quiet:
            print("Auto-detected splits:", list(masks.keys()))

        # Check masks are all the same length.
        lengths = set(x.shape[0] for x in masks.values())
        if len(lengths) == 0:
            raise ValueError("No split masks found.")
        elif len(lengths) != 1:
            raise ValueError(f"Mask arrays must all be the same size, got sizes: {lengths}")
        n_sys = list(lengths)[0]

        # Check that masks define a complete split
        mask_counts = torch.zeros(n_sys, dtype=int)
        for k, arr in masks.items():
            mask_counts += arr.to(int)
        if not (mask_counts == 1).all():
            set_of_counts = set(mask_counts)
            raise ValueError(
                f" Auto-splitting requires unique split for each item."
                + f" Items with the following split counts were detected: {set_of_counts}"
            )

        if dry_run:
            return

        masks = {k: self.arr_dict["indices"][m] for k, m in masks.items()}
        for k, m in masks.items():
            self.make_explicit_split(k, m)

        if not self.quiet:
            print("Finished automatic splitting.")

        assert arrdict_len(self.arr_dict) == 0, "Not all items were successfully auto-split."

        self.splitting_completed = True

        return

    def make_generator(self,
                       split_name: str,
                       evaluation_mode: str,
                       batch_size: Union[int, None] = None,
                       subsample: Union[float, bool] = False
                       ):
        """
        Makes a dataloader for the given type of split and evaluation mode of the model.

        In most cases, you do not need to call this function directly as a user.

        :param split_name: str; "train", "valid", or "test" ; selects data to use
        :param evaluation_mode: str; "train" or "eval". Used for whether to shuffle.
        :param batch_size: passed to pytorch
        :param subsample: fraction to subsample
        :return: dataloader containing relevant data

        """
        if not self.splitting_completed:
            raise ValueError("Database has not yet been split.")

        if split_name not in self.splits:
            raise ValueError(f"Split {split_name} Invalid. Current splits:{list(self.splits.keys())}")

        data = [self.splits[split_name][k] for k in self.var_list]

        if evaluation_mode == "train":
            if split_name != "train":
                raise ValueError("evaluation mode 'train' can only be used with training data." "(got {})".format(split_name))
            shuffle = True
        elif evaluation_mode == "eval":
            shuffle = False
        else:
            raise ValueError(f"Evaluation_mode ({evaluation_mode}) must be one of 'train' or 'eval'")

        dataset = NamedTensorDataset(self.var_list, *data)
        if subsample:
            n_total = data[0].shape[0]
            n_selected = int(n_total * subsample)
            sampled_indices = torch.argsort(torch.rand(n_total))[:n_selected]
            dataset = Subset(dataset, sampled_indices)

        generator = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=sparse_enabled_colate,
            **self.dataloader_kwargs,
        )

        return generator

    def _array_stat_helper(self, key, species_key, atomwise, norm_per_atom, norm_axis):

        prop = self.arr_dict[key]

        if norm_axis:
            prop = torch.norm(prop, dim=norm_axis)

        if atomwise:
            if norm_per_atom:
                raise ValueError("norm_per_atom and atom_var cannot both be True!")
            if species_key is None:
                raise RuntimeError("species_key must be given to trim a atomwise quantity")

            real_atoms = self.arr_dict[species_key] > 0
            stat_prop = prop[real_atoms]
        else:
            stat_prop = prop

        if norm_per_atom:
            if species_key is None:
                raise RuntimeError("species_key must be given to trim an atom-normalized quantity")

            n_atoms = (self.arr_dict[species_key] > 0).sum(dim=1)
            # Transposes broadcast the result rightwards instead of leftwards.
            # numpy transpose on higher-order arrays reverses all dimensions.
            prop = (prop.T / n_atoms).T
            stat_prop = (stat_prop.T / n_atoms).T

        mean = stat_prop.mean()
        std = stat_prop.std()

        if mean.dtype.is_floating_point and torch.isnan(mean).item():
            has_nan = True
        elif std.dtype.is_floating_point and torch.isnan(std).item():
            has_nan = True
        else:
            has_nan = False

        if has_nan:
            warnings.warn(f"Array statistics, {mean=},{std=} contain NaN.", stacklevel=3)

        return prop, mean, std

    def remove_high_property(
        self,
        key: str,
        atomwise: bool,
        norm_per_atom: bool = False,
        species_key: str = None,
        cut: Union[float, None] = None,
        std_factor: Union[float, None] = 10,
        norm_axis: Union[int, None] = None,
    ):
        """
        For removing outliers from a dataset. Use with caution; do not inadvertently remove outliers from benchmarks!

        The parameters cut and std_factor can be set to `None` to avoid their steps.
        the per_atom and atom_var properties are exclusive; they cannot both be true.

        :param key: The property key in the dataset to check for high values
        :param atomwise: True if the property is defined per atom in axis 1, otherwise property is treated as whole-system value
        :param norm_per_atom: True if the property should be normalized by atom counts
        :param species_key: Which array represents the atom presence; required if per_atom is True
        :param cut: If values > mu + cut, the system is removed. The step done first.
        :param std_factor: If (value-mu)/std > std_fact, the system is trimmed. This step done second.
        :param norm_axis: if not None, the property array is normed on the axis. Useful for vector properties like force.
        :return:
        """
        print(f"Cutting on variable: {key}")
        if cut is not None:
            prop, mean, std = self._array_stat_helper(key, species_key, atomwise, norm_per_atom, norm_axis)

            large_property_mask = torch.abs(prop - mean) > cut
            # Scan over all non-batch indices.
            non_batch_axes = tuple(range(1, prop.ndim))
            drop_mask = torch.sum(large_property_mask, dim=non_batch_axes) > 0
            indices = self.arr_dict["indices"][drop_mask]
            if drop_mask.any():
                print(f"Removed {drop_mask.to(int).sum()} outlier systems in variable {key} due to static cut.")
                self.make_explicit_split(f"failed_cut_{key}", indices)

        if std_factor is not None:
            prop, mean, std = self._array_stat_helper(key, species_key, atomwise, norm_per_atom, norm_axis)
            large_property_mask = torch.abs(prop - mean) / std > std_factor
            # Scan over all non-batch indices.
            non_batch_axes = tuple(range(1, prop.ndim))
            drop_mask = torch.sum(large_property_mask, dim=non_batch_axes) > 0
            indices = self.arr_dict["indices"][drop_mask]
            if drop_mask.any():
                print(f"Removed {drop_mask.to(int).sum()} outlier systems in variable {key} due to std. factor.")
                self.make_explicit_split(f"failed_std_fac_{key}", indices)

    def write_h5(self,
                 split: Union[str, None] = None,
                 h5path: Union[str, None] = None,
                 species_key: str = "species",
                 overwrite:bool = False):
        """
        Write this database to the pyanitools h5 format.
        See :func:`hippynn.databases.h5_pyanitools.write_h5` for details.

        Note: This function will error if h5py is not installed.

        :param split:
        :param h5path:
        :param species_key:
        :param overwrite:
        :return:
        """

        try:
            from .h5_pyanitools import write_h5 as write_h5_function
        except ImportError as ie:
            raise ImportError("Writing h5 versions of databases not available.") from ie

        return write_h5_function(self, split=split, file=h5path, species_key=species_key, overwrite=overwrite)

    def write_npz(
        self,
        file: str,
        record_split_masks: bool = True,
        compressed: bool = True,
        overwrite: bool = False,
        split_prefix: Union[str, None] = None,
        return_only: bool = False,
    ):
        """
        :param file: str, Path, or file object compatible with np.save
        :param record_split_masks: whether to generate and place masks for the splits into the saved database.
        :param compressed: whether to use np.savez_compressed (True) or np.savez
        :param overwrite: Whether to accept an existing path. Only used if fname is str or path.
        :param split_prefix: optionally override the prefix for the masks computed by the splits.
        :param return_only: if True, ignore the file string and just return the resulting dictionary of numpy arrays.
        :return:
        """
        if split_prefix is None:
            split_prefix = _AUTO_SPLIT_PREFIX
        if not self.splitting_completed:
            raise ValueError(
                "Cannot write an incompletely split database to npz file.\n"
                + "You can split the rest using `database.split_the_rest('other_data')`\n"
                + "to put the remaining data into a new split named 'other_data'"
            )

        # get combined dictionary of arrays.
        np_dict = {sname: {arr_name: array.to("cpu").numpy() for arr_name, array in split.items()} for sname, split in self.splits.items()}

        # insert split masks if requested.

        if record_split_masks:
            self.add_split_masks(dict_to_add_to=np_dict, split_prefix=split_prefix)

        # Stack numpy arrays:
        arr_dict = {}
        a_split = list(np_dict.values())[0]
        keys = a_split.keys()

        for k in list(keys):
            list_of_arrays = [split_dict[k] for split_dict in np_dict.values()]
            arr_dict[k] = np.concatenate(list_of_arrays, axis=0)

        # Put results where requested.
        if return_only:
            return arr_dict

        if isinstance(file, str):
            file = Path(file)

        if isinstance(file, Path):
            if file.exists() and not overwrite:
                raise FileExistsError(f"File exists: {file}")

        if compressed:
            np.savez_compressed(file, **arr_dict)
        else:
            np.savez(file, **arr_dict)

        return arr_dict

    def sort_by_index(self, index_name: str = "indices"):
        """

        Sort arrays in each split of the database by an index key.

        The default is 'indices', also possible is 'split_indices', or any other variable name in the database.

        :param index_name:
        :return: None
        """
        for sname, split in self.splits.items():
            ind = split[index_name]
            ind_order = torch.argsort(ind)
            # Modify dictionary in-place.
            for k, v in split.items():
                split[k] = v[ind_order]

    def trim_by_species(self, species_key: str, keep_splits_same_size: bool = True):
        """
        Remove any excess padding in a database.

        :param species_key: what array to use to mark atom presence.
        :param keep_splits_same_size: true: trim by the minimum amount across splits,
          false:  trim by the maximum amount for each split.
        :return: None
        """
        if not self.splitting_completed:
            raise ValueError("Cannot trim arrays until splitting has been completed.")

        split_max_max_atom_size = {}
        for k, split in self.splits.items():
            species_array = split[species_key]
            max_atoms = (species_array != 0).sum(dim=1)
            max_max_atoms = max_atoms.max().item()
            split_max_max_atom_size[k] = max_max_atoms
            del max_atoms, species_array, max_max_atoms  # Marking unneeded.

        if keep_splits_same_size:
            # find the longest of the split sizes
            max_max_max_atoms = max(split_max_max_atom_size.values())
            # store that back into the dictionary
            split_max_max_atom_size = {k: max_max_max_atoms for k, v in split_max_max_atom_size.items()}
            del max_max_max_atoms  # Marking unneeded.

        for k, split in self.splits.items():
            species_array = split[species_key]
            orig_atom_size = species_array.shape[1]
            max_max_atoms = split_max_max_atom_size[k]
            order = torch.argsort(species_array, dim=1, descending=True, stable=True)

            assert max_max_atoms > 7, "Max atoms bigger than 7 required for automatic atom dimension detection."

            for key, arr in split.items():

                # determine where to broadcast sorting indices for this array.
                non_species_non_batch_axes = []
                for dim, length in enumerate(arr.shape[1:], start=1):
                    if length == orig_atom_size:
                        pass
                    else:
                        non_species_non_batch_axes.append(dim)

                for dim, length in enumerate(arr.shape[1:], start=1):
                    if dim in non_species_non_batch_axes:
                        continue

                    unsq_dims = tuple(x for x in non_species_non_batch_axes if x != dim)
                    this_order = unsqueeze_multiple(order, unsq_dims)
                    arr, this_order = torch.broadcast_tensors(arr, this_order)
                    arr = torch.take_along_dim(arr, this_order, dim)
                    arr = torch.narrow_copy(arr, dim, 0, max_max_atoms)
                    if not self.quiet:
                        print(f"Resorting {key} along axis {dim}. {arr.shape=},{this_order.shape=}")

                split[key] = arr
                # end loop over arrays
            # end loop over splits

        return

    def get_device(self) -> torch.device:
        """
        Determine what device the database resides on. Raises ValueError if multiple devices are encountered.

        :return: device.
        """
        if not self.splitting_completed:
            raise ValueError("Device should not be changed before splitting is complete.")

        devices = set(a.device for s, split in self.splits.items() for k, a in split.items())
        if len(devices) != 1:
            raise ValueError(f"Devices for tensors are not uniform, got: {devices}")

        device = devices.pop()
        return device

    def make_database_cache(self, file: str = "./hippynn_db_cache.npz", overwrite: bool = False, **override_kwargs) -> "Database":
        """
        Cache the database as-is, and re-open it.

        Useful for creating an easy restart script if the storage space is available.
        The new datatbase will by default inherit the properties of this database.

        usage:
        >>> database = database.make_database_cache()

        :param file: where to store the database
        :param overwrite: whether to overwrite an existing cache file with this name.
        :param override_kwargs: passed to NPZDictionary instead of the current database settings.
        :return: The new database created from the cache.
        """
        from .ondisk import NPZDatabase

        # first prepare arguments
        arguments = dict(
            file=file,
            inputs=self.inputs,
            targets=self.targets,
            seed=self.random_state.get_state(),
            test_size=None,  # using auto_split
            valid_size=None,  # using_auto_split
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            allow_unfound=True,  # We may have extra arrays; reproduce them.
            auto_split=True,  # Inherit splitting from this db.
            device=self.get_device(),
            quiet=self.quiet,
        )

        if override_kwargs:
            if not self.quiet:
                print("Overriding arguments to database cache:", override_kwargs)
            arguments.update(override_kwargs)

        # now write cache
        if not self.quiet:
            print("Writing Cached database to", file)

        self.write_npz(
            file=file, record_split_masks=True, overwrite=overwrite, return_only=False  # allows inheriting of splits from this db.
        )
        # now reload cached file.
        return NPZDatabase(**arguments)


def compute_index_mask(indices: torch.Tensor, index_pool: torch.Tensor) -> torch.Tensor:
    """

    :param indices:
    :param index_pool:
    :return:
    """
    index_set = set(index_pool.tolist())
    if not all(i.item() in index_set for i in indices):
        raise ValueError("Provided indices not in database")

    uniques, counts = torch.unique(indices, return_counts=True)
    if uniques.numel() != indices.numel():
        raise ValueError("Split indices not unique")
    if counts.max() > 1:
        raise ValueError("Split indices have duplicates.")

    index_mask = torch.isin(index_pool, indices)
    return index_mask


def prettyprint_arrays(arr_dict: dict[str: torch.Tensor]):
    """
    Pretty-print array dictionary.
    :return: None
    """
    column_format = "| {:<30} | {:<18} | {:<28} |"
    ncols = len(column_format.format("", "", ""))

    def printrow(*args):
        print(column_format.format(*args))

    def printline():
        print("-" * ncols)

    printline()
    printrow("Name", "dtype", "shape")
    printline()
    for key, value in arr_dict.items():
        if isinstance(value, torch.Tensor):
            value = value.numpy()
        printrow(key, repr(value.dtype), repr(value.shape))
    printline()


class NamedTensorDataset(TensorDataset):
    def __init__(self, tensor_names, *tensors):
        super().__init__(*tensors)
        self.tensor_map = tensor_names


collate_map = defaultdict(torch.utils.data.default_collate)

def tensor_collate(list_of_tensors, collate_fn_map=None):
    elem = list_of_tensors[0]
    if elem.layout == torch.strided:
        return torch.utils.data.default_collate(list_of_tensors)
    return torch.stack(list_of_tensors, dim=0)
collate_map[torch.Tensor] = tensor_collate

def sparse_enabled_colate(batch):
    return torch.utils.data._utils.collate.collate(batch, collate_fn_map=collate_map)
