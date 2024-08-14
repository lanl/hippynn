"""
Based on pyanitools.py written by Roman Zubatyuk and Justin S. Smith:
https://github.com/atomistic-ml/ani-al/blob/master/readers/lib/pyanitools.py
"""

import os
import numpy as np
import h5py


class DataPacker:
    def __init__(self, store_file, mode='w-', compression_lib='gzip', compression_level=6, driver=None):
        """
        Wrapper to store arrays within HFD5 file
        """
        self.store = h5py.File(store_file, mode=mode, driver=driver)
        self.compression = compression_lib
        self.compression_opts = compression_level

    def store_data(self, store_location, **kwargs):
        """
        Put arrays to store
        """
        group = self.store.create_group(store_location)

        for name, data in kwargs.items():
            if isinstance(data, list):
                if len(data) != 0:
                    if type(data[0]) is np.str_ or type(data[0]) is str:
                        data = [a.encode('utf8') for a in data]

            group.create_dataset(name, data=data, compression=self.compression, compression_opts=self.compression_opts)

    def cleanup(self):
        """
        Wrapper to close HDF5 file
        """
        self.store.close()

    def __del__(self):
        if self.store is not None:
            self.cleanup()


class AniDataLoader(object):
    def __init__(self, store_file, driver=None):
        """
        Constructor
        """
        if not os.path.exists(store_file):
            store_file = os.path.realpath(store_file)
            self.store = None
            raise FileNotFoundError(f'File not found: {store_file}')
        self.store = h5py.File(store_file, driver=driver)

    def h5py_dataset_iterator(self, g, prefix=''):
        """
        Group recursive iterator (iterate through all groups in all branches and return datasets in dicts)
        """

        for key, item in g.items():

            path = f'{prefix}/{key}'

            first_subkey = list(item.keys())[0]
            first_subitem = item[first_subkey]

            if isinstance(first_subitem, h5py.Dataset):
                # If dataset, yield the data from it.
                data = self.populate_data_dict({'path': path}, item)
                yield data
            else:
                # If not a dataset, assume it's a group and iterate from that.
                yield from self.h5py_dataset_iterator(item, path)

    def __iter__(self):
        """
        Default class iterator (iterate through all data)
        """
        for data in self.h5py_dataset_iterator(self.store):
            yield data

    def get_group_list(self):
        """
        Returns a list of all groups in the file
        """
        return [g for g in self.store.values()]

    def iter_group(self, g):
        """
        Allows interation through the data in a given group
        """
        for data in self.h5py_dataset_iterator(g):
            yield data

    def get_data(self, path, prefix=''):
        """
        Returns the requested dataset
        """
        item = self.store[path]
        data = self.populate_data_dict({'path': f'{prefix}/{path}'}, item)

        return data

    @staticmethod
    def populate_data_dict(data, group):
        for key, value in group.items():

            if not isinstance(value, h5py.Group):
                dataset = np.asarray(value[()])

                # decode bytes objects to ascii strings.
                if isinstance(dataset, np.ndarray):
                    if dataset.size != 0:
                        if type(dataset[0]) is np.bytes_:
                            dataset = [a.decode('ascii') for a in dataset]

                data.update({key: dataset})

        return data

    def group_size(self):
        """
        Returns the number of groups
        """
        return len(self.get_group_list())

    def size(self):
        count = 0
        for g in self.store.values():
            count = count + len(g.items())
        return count

    def cleanup(self):
        """
        Close the HDF5 file
        """
        self.store.close()

    def __del__(self):
        if self.store is not None:
            self.cleanup()
