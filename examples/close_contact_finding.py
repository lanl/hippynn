"""
close_contact_finding.py

This example shows how to use the hippynn function calculate_min_dists
to find distances in the dataset where there are close contacts.
Such a procedure is often useful in active learning to identify
outlier data which inhibits training.

This script was designed for an external dataset available at
https://github.com/atomistic-ml/ani-al

Note: It is necessary to untar the h5 data files in ani-al/data/
before running this script.

"""
import sys

sys.path.append("../../datasets/ani-al/readers/lib/")
import pyanitools  # Check if pyanitools is found early

### Loading the database
from hippynn.databases.h5_pyanitools import PyAniDirectoryDB

database = PyAniDirectoryDB(
    directory="../../datasets/ani-al/data/",
    seed=0,
    quiet=False,
    allow_unfound=True,
    inputs=None,
    targets=None,
)

### Calculating minimum distance array
from hippynn.pretraining import calculate_min_dists

min_dist_array = calculate_min_dists(
    database.arr_dict,
    species_name="species",
    positions_name="coordinates",
    cell_name="cell",  # for open boundaries, do not pass a cell name (or pass None)
    dist_hard_max=4.0,  # check for distances up to this radius
    batch_size=50,
)

print("Minimum distance in configurations:")
print(f"{min_dist_array.dtype=}")
print(f"{min_dist_array.shape=}")
print("First 100 values:", min_dist_array[:100])

### Making a plot of the minimum distance for each configuration
import matplotlib.pyplot as plt

plt.hist(min_dist_array, bins=100)
plt.title("Minimum distance per config")
plt.xlabel("Distance")
plt.ylabel("Count")
plt.yscale("log")
plt.show()

#### How to remove and separate low distance configurations
dist_thresh = 1.7  # Note: what threshold to use may be highly problem-dependent.
low_dist_configs = min_dist_array < dist_thresh
where_low_dist = database.arr_dict["indices"][low_dist_configs]

# This makes the low distance configurations
# into their own split, separate from train/valid/test.
database.make_explicit_split("LOW_DISTANCE_FILTER", where_low_dist)

# This deletes the new split, although deleting it is not necessary;
# this data will not be included in train/valid/test splits
del database.splits["LOW_DISTANCE_FILTER"]

### Continue on with data processing, e.g.
database.make_trainvalidtest_split(test_size=0.1, valid_size=0.1)
