"""
Base database functionality from dictionary of numpy arrays
"""
import warnings
import numpy as np
import torch

from .restarter import NoRestart
from ..tools import arrdict_len, device_fallback

from torch.utils.data import DataLoader, TensorDataset, Subset


class Database:
    """
    Class for holding a pytorch dataset, splitting it, generating dataloaders, etc."
    """

    def __init__(
        self,
        arr_dict,
        inputs,
        targets,
        seed,
        test_size=None,
        valid_size=None,
        num_workers=0,
        pin_memory=True,
        allow_unfound=False,
        quiet=False,
    ):
        """
        :param arr_dict: dictionary mapping strings to numpy arrays
        :param inputs:   list of strings for input db_names
        :param targets:  list of strings for output db_namees
        :param seed:     int, for random splitting
        :param test_size: fraction of data to use in test split
        :param valid_size: fraction of data to use in train split
        :param num_workers: passed to pytorch dataloaders
        :param pin_memory: passed to pytorch dataloaders
        :param allow_unfound: If true, skip checking if the needed inputs and targets are found.
           This allows setting inputs=None and/or targets=None.
        :param quiet: If True, print little or nothing while loading.
        """

        # Restartable Children of this class should change this after super().__init__ .
        self.restarter = NoRestart()

        self.inputs = inputs
        self.targets = targets
        self.quiet = quiet
        self.splitting_completed = False
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if not quiet:
            print(f"All arrays:")
            prettyprint_arrays(arr_dict)

        for k in self.var_list:
            if k not in arr_dict and k not in ("indices", "split_indices"):
                if allow_unfound:
                    warnings.warn(f"Required database quantity '{k}' not present.")
                else:
                    raise KeyError(
                        f"Array dictionary missing required variable:'{k}'."
                        "Pass allow_unfound=True to avoid checking of inputs targets."
                    )

        self.arr_dict = arr_dict
        if "indices" not in arr_dict:
            if not quiet:
                print("Database: Using auto-generated data indices")
            self.arr_dict["indices"] = np.arange(len(self), dtype=int)
        else:
            if not quiet:
                print("Database: Using pre-specified data indices.")

        self.splits = {}
        self.random_state = np.random.RandomState(seed=seed)

        if test_size is not None or valid_size is not None:
            if test_size is None or valid_size is None:
                raise ValueError("Both test and valid size must be set for auto-splitting")
            else:
                self.make_trainvalidtest_split(test_size=test_size, valid_size=valid_size)

    def __len__(self):
        return arrdict_len(self.arr_dict)

    @property
    def var_list(self):
        if self.inputs is None:
            raise RuntimeError(f"Database inputs not defined, set {Database}.inputs.")
        if self.targets is None:
            raise RuntimeError(f"Database inputs not defined, set {Database}.targets.")
        return self.inputs + self.targets

    def send_to_device(self, device=None):
        "Send a database to a device"
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

    def make_random_split(self, evaluation_mode, split_size):
        """

        :param evaluation_mode: String naming the split, can be anything, but 'train', 'valid', and 'test' are special.s
        :param split_size: int (number of items) or float<1, fraction of samples.
        :return:
        """
        if self.splitting_completed:
            raise RuntimeError("Database already split!")

        if split_size < 1:
            split_size = int(split_size * len(self))

        split_indices = self.random_state.choice(self.arr_dict["indices"], size=split_size, replace=False)

        split_indices.sort()

        return self.make_explicit_split(evaluation_mode, split_indices)

    def make_trainvalidtest_split(self, test_size, valid_size):
        if self.splitting_completed:
            raise RuntimeError("Database already split!")

        if valid_size < 1:
            if test_size >= 1:
                raise ValueError("If train or valid size is set as a fraction, then set test_size as a fraction")
            else:
                if valid_size + test_size > 1:
                    raise ValueError(
                        f"Test fraction ({test_size}) plus valid fraction " f"({valid_size}) are greater than 1!"
                    )
                valid_size /= 1 - test_size

        self.make_random_split("test", test_size)
        self.make_random_split("valid", valid_size)
        self.split_the_rest("train")

    def make_explicit_split(self, evaluation_mode, split_indices):
        if self.splitting_completed:
            raise RuntimeError("Database already split!")

        if len(split_indices) == 0:
            raise ValueError("Cannot make split of size 0.")
        # Compute which indices are not being split off.
        index_mask = compute_index_mask(split_indices, self.arr_dict["indices"])

        complement_mask = ~index_mask

        # Split off data, and keep the rest.
        self.splits[evaluation_mode] = {k: torch.from_numpy(self.arr_dict[k][index_mask]) for k in self.arr_dict}
        self.splits[evaluation_mode]["split_indices"] = torch.arange(len(split_indices), dtype=torch.int64)

        for k, v in self.arr_dict.items():
            self.arr_dict[k] = v[complement_mask]

        if not self.quiet:
            print(f"Arrays for split: {evaluation_mode}")
            prettyprint_arrays(self.splits[evaluation_mode])

    def split_the_rest(self, evaluation_mode):
        self.make_explicit_split(evaluation_mode, self.arr_dict["indices"])
        self.splitting_completed = True

    def make_generator(self, split_type, evaluation_mode, batch_size=None, subsample=False):
        """
        Makes a dataloader for the given type of split and evaluation mode of the model.

        :param split_type: str; "train", "valid", or "test" ; selects data to use
        :param evaluation_mode: str; "train" or "eval". Used for whether to shuffle.
        :param batch_size: passed to pytorch
        :param subsample: fraction to subsample
        :return: dataloader containing relevant data

        """
        if not self.splitting_completed:
            raise ValueError("Database has not yet been split.")

        if split_type not in self.splits:
            raise ValueError(f"Split {split_type} Invalid. Current splits:{list(self.splits.keys())}")

        data = [self.splits[split_type][k] for k in self.var_list]

        if evaluation_mode == "train":
            if split_type != "train":
                raise ValueError(
                    "evaluation mode 'train' can only be used with training data." "(got {})".format(split_type)
                )
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
            # sampled_indices = torch.rand(data[0].shape[0]) < subsample
            dataset = Subset(dataset, sampled_indices)
            # data = [a[sampled_indices] for a in data]

        generator = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

        return generator
        
    def trim_all_arrays(self,index):
        """
        To be used in conjuction with remove_high_property
        """
        for key in self.arr_dict:
            self.arr_dict[key] = self.arr_dict[key][index]
    
    def remove_high_property(self,key,perAtom,species_key=None,cut=None,std_factor=10):
        """
        This function removes outlier data from the dataset
        Must be called before splitting
        "key": the property key in the dataset to check for high values
        "perAtom": True if the property is defined per atom in axis 1, otherwise property is treated as full system
        "std_factor": systems with values larger than this multiplier time the standard deviation of all data will be reomved. None to skip this step
        "cut_factor": systems with values larger than this number are reomved. None to skip this step. This step is done first. 
        """
        if perAtom:
            if species_key==None:
                raise RuntimeError("species_key must be defined to trim a per atom quantity")
            atom_ind = self.arr_dict[species_key] > 0
        ndim = len(self.arr_dict[key].shape)
        if cut!=None:
            if perAtom:
                Kmean = np.mean(self.arr_dict[key][atom_ind])
            else:
                Kmean = np.mean(self.arr_dict[key])
            failArr = np.abs(self.arr_dict[key]-Kmean)>cut
            #This does nothing with ndim=1
            trimArr = np.sum(failArr,axis=tuple(range(1,ndim)))==0
            self.trim_all_arrays(trimArr)
            
        if std_factor!=None:
            if perAtom:
                atom_ind = self.arr_dict[species_key] > 0
                Kmean = np.mean(self.arr_dict[key][atom_ind])
                std_cut = np.std(self.arr_dict[key][atom_ind]) * std_factor
            else: 
                Kmean = np.mean(self.arr_dict[key])
                std_cut = np.std(self.arr_dict[key]) * std_factor
            failArr = np.abs(self.arr_dict[key]-Kmean)>std_cut
            #This does nothing with ndim=1
            trimArr = np.sum(failArr,axis=tuple(range(1,ndim)))==0
            self.trim_all_arrays(trimArr)



def compute_index_mask(indices, index_pool):
    if not np.all(np.isin(indices, index_pool)):
        raise ValueError("Provided indices not in database")

    uniques, counts = np.unique(indices, return_counts=True)
    if len(uniques) != len(indices):
        raise ValueError("Split indices not unique")
    if counts.max() > 1:
        raise ValueError("Split indices have duplicates.")

    index_mask = np.isin(index_pool, indices)
    return index_mask


def prettyprint_arrays(arr_dict):
    """
    Pretty-print array dictionary
    :return: None
    """
    column_format = "| {:<18} | {:<18} | {:<40} |"
    ncols = len(column_format.format("", "", ""))

    def printrow(*args):
        print(column_format.format(*args))

    def printline():
        print("-" * ncols)

    printline()
    printrow("Name", "dtype", "shape")
    printline()
    for key, value in arr_dict.items():
        printrow(key, repr(value.dtype), repr(value.shape))
    printline()


class NamedTensorDataset(TensorDataset):
    def __init__(self, tensor_names, *tensors):
        super().__init__(*tensors)
        self.tensor_map = tensor_names
