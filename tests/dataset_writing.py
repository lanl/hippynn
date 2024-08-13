import glob
import torch

torch.set_default_dtype(torch.float64)

from hippynn.databases.h5_pyanitools import PyAniFileDB
from hippynn.databases import NPZDatabase
from hippynn.tools import is_equal_state_dict


# compare if databases are equal, split by split
def eqsplit(db1, db2, raise_error=True):
    return is_equal_state_dict(db1.splits, db2.splits, raise_where=raise_error)


if __name__ == "__main__":
    CLEANUP = True  # delete datasets afterwards
    # Example dataset
    location = "../../datasets/new_qm9_clean.npz"
    seed = 1
    num_workers = 0
    db_info = {}
    db1 = NPZDatabase(
        file=location,
        seed=seed,
        num_workers=num_workers,
        allow_unfound=True,
        **db_info,
        inputs=None,
        targets=None,
    )

    # test remove_high_property
    db1.remove_high_property("E", species_key="Z", atomwise=False, norm_per_atom=True, std_factor=5)

    # throw stuff away
    db1.make_random_split("random stuff", 0.99)
    del db1.splits["random stuff"]  # remove something at random

    new_ani_file = "TEST_clean_ani1x.h5"
    new_npz_file = "TEST_clean_ani1x.npz"

    # Divide up the dataset in a bunch of splits.
    db1.make_random_split("first", 0.5)
    db1.make_random_split("second", 0.2)
    db1.make_random_split("third", 3)  # integer
    db1.split_the_rest("remaining")
    # write an npz version and reload it.
    db1.write_npz(file=new_npz_file, record_split_masks=True, overwrite=True)
    db3 = NPZDatabase(file=new_npz_file, seed=seed, num_workers=num_workers, allow_unfound=True, inputs=None, targets=None, auto_split=True, **db_info)

    # write an h5 version and reload it.
    db1.write_h5(split=True, h5path=new_ani_file, species_key="Z", overwrite=True)
    db2 = PyAniFileDB(
        file=new_ani_file,
        species_key="Z",
        seed=seed,
        num_workers=num_workers,
        allow_unfound=True,
        **db_info,
        inputs=None,
        targets=None,
        auto_split=True,
    )
    new_ani_filetwo = "TEST_clean_ani1x_2.h5"
    # trim this dataset earlier than others.
    db2.trim_by_species("Z")
    # write and load new database.
    db2.write_h5(split=True, h5path=new_ani_filetwo, species_key="Z", overwrite=True)
    db4 = PyAniFileDB(
        file=new_ani_filetwo,
        species_key="Z",
        seed=seed,
        num_workers=num_workers,
        allow_unfound=True,
        **db_info,
        inputs=None,
        targets=None,
        auto_split=True,
    )

    for i, d in enumerate([db1, db2, db3, db4]):
        print("sorting", i)
        d.sort_by_index()
        print("trimming", i)
        d.trim_by_species("Z", keep_splits_same_size=True)  # can do either way if disable caching test.

    # "sys_number" comes from h5 format databases, but not present in others.
    for d in [db2, db4]:
        for s in d.splits:
            del d.splits[s]["sys_number"]

    db1.add_split_masks()  # this first didn't ever get split masks! add them now.

    print("NPZ Equality:", eqsplit(db1, db3))
    print("H5 Equality:", eqsplit(db2, db4))
    print("NPZ-H5 Equality1:", eqsplit(db1, db2))
    print("NPZ-H5 Equality2:", eqsplit(db2, db3))

    # test caching routine.
    db2p = db2.make_database_cache(file="TEST_CACHE_FROMH5.npz", overwrite=True, quiet=True)
    print("H5 to cache Equality:", eqsplit(db2, db2p))
    db3p = db3.make_database_cache(file="TEST_CACHE_FROMNPZ.npz", overwrite=True, quiet=True)
    print("NPZ to cache Equality:", eqsplit(db3, db3p))

    for ext in ["npz", "h5"]:

        for file in glob.glob(f"./TEST*.{ext}"):
            print(file)
            import os

            os.remove(file)
