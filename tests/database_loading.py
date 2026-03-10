from pathlib import Path
import pytest

from ase.io import read

from hippynn.databases import AseDatabase, AseDatabaseIterable

DATA_DIR = Path(__file__).parents[2] / "datasets"

# both files contain the first 70 lines from methane.extxyz, available here:
# https://archive.materialscloud.org/records/kz78r-6nx43
# $ head -n 70 methane.extxyz > methane_small.extxyz
methane_small_1 = "methane_small.extxyz"
methane_small_2 = "methane_small2.extxyz"


@pytest.mark.xfail(not (DATA_DIR / methane_small_1).exists(), reason="Data file not available", strict=True)
def test_load_AseDatabase():
    database = AseDatabase(
        directory=DATA_DIR,
        name=methane_small_1,
        inputs=[],
        targets=[],
        allow_unfound=True,
        seed=0,
    )

    expected_keys = {"numbers", "positions", "cell", "forces", "energy"}
    assert expected_keys.issubset(database.arr_dict.keys())

    assert database.arr_dict["numbers"].shape == (10, 5)


@pytest.mark.xfail(
    not ((DATA_DIR / methane_small_1).exists() and (DATA_DIR / methane_small_2).exists()), reason="Data files not al available", strict=True
)
def test_load_multiple_AseDatabase():
    database = AseDatabase(
        directory=DATA_DIR,
        name=[methane_small_1, methane_small_2],
        inputs=[],
        targets=[],
        allow_unfound=True,
        seed=0,
    )

    assert database.arr_dict["numbers"].shape == (20, 5)


@pytest.mark.xfail(not (DATA_DIR / methane_small_1).exists(), reason="Data file not available", strict=True)
def test_load_AseDatabaseIterable():
    database = AseDatabaseIterable(
        iterable=read(DATA_DIR / methane_small_1, index=slice(0, 5)),
        inputs=[],
        targets=[],
        allow_unfound=True,
        seed=0,
    )

    assert database.arr_dict["numbers"].shape == (5, 5)
