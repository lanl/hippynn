import pytest
import torch

from hippynn.databases.database import Database


@pytest.fixture
def dummy_db() -> Database:
    arr_dict = {"a": torch.randn(5, 1), "b": torch.randn(5, 1)}
    return Database(
        arr_dict=arr_dict,
        inputs=["a"],
        targets=["b"],
        seed=0,
        quiet=True,
    )


def test_align_database_success(dummy_db: Database) -> None:
    """When the provided lists match the database sets, ordering is updated."""
    dummy_db.align(["a"], ["b"])
    assert dummy_db.inputs == ["a"]
    assert dummy_db.targets == ["b"]


def test_align_database_failure(dummy_db: Database) -> None:
    """A mismatched input or target set should raise ``ValueError``."""
    with pytest.raises(ValueError):
        dummy_db.align(["c"], ["b"])

    with pytest.raises(ValueError):
        dummy_db.align(["a"], ["c"])

