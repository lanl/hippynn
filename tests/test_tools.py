import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

import hippynn
from hippynn.tools import active_directory, log_terminal

from conftest import ignore_weights_only_warning, ignore_pin_memory_mps_warning, ignore_sensitivity_warning


def test_active_directory(temporary_directory):
    cwd = Path.cwd()
    assert active_directory == cwd
    with active_directory("outer") as outer, active_directory("inner") as inner:
        assert inner == cwd / "outer" / "inner" and inner.is_dir()
        assert active_directory / "f.pt" == inner / "f.pt"
        assert Path.cwd() == cwd
        with active_directory(None) as same:
            assert same == inner

        seen = []
        thread = threading.Thread(target=lambda: seen.append(active_directory.resolve()))
        thread.start()
        thread.join()
        assert seen == [cwd]

    with active_directory("outer", chdir=True):
        assert Path.cwd() == outer
    assert Path.cwd() == cwd

    with pytest.raises(FileNotFoundError):
        with active_directory("missing", create=False):
            pass
    with pytest.raises(FileExistsError):
        with active_directory("outer", create=True):
            pass
    with pytest.raises(TypeError, match="resolve"):
        hash(active_directory)
    assert active_directory == cwd


def test_log_terminal(temporary_directory):
    stdout, stderr = sys.stdout, sys.stderr
    with pytest.raises(ValueError):
        with active_directory("exp"), log_terminal("outer.txt", "wt"):
            print("one")
            with log_terminal("inner.txt", "wt"):
                print("two", file=sys.stderr)
                thread = threading.Thread(target=lambda: print("other-thread"))
                thread.start()
                thread.join()
            raise ValueError("logged-error")
    assert sys.stdout is stdout and sys.stderr is stderr

    outer, inner = Path("exp/outer.txt").read_text(), Path("exp/inner.txt").read_text()
    assert "one" in outer and "two" in outer and "logged-error" in outer
    assert "two" in inner and "one" not in inner
    assert "other-thread" not in outer


@ignore_weights_only_warning
@ignore_pin_memory_mps_warning
@ignore_sensitivity_warning
def test_train_in_directory(temporary_directory, example_training_setup):
    from hippynn.databases import NPZDatabase

    training_modules, database, setup_params = example_training_setup

    # Relative dataset paths are not affected by the active directory (issue #202).
    np.savez("data.npz", **{k: v.numpy() for k, v in database.arr_dict.items()})
    with active_directory("run1"), log_terminal("training_log.txt", "wt"):
        NPZDatabase("data.npz", database.inputs, database.targets, seed=0, quiet=True)
        hippynn.setup_and_train(training_modules, database, setup_params)

    for fname in ["experiment_structure.pt", "best_model.pt", "best_checkpoint.pt", "training_log.txt"]:
        assert Path("run1", fname).exists(), fname
    assert not Path("best_model.pt").exists()

    with active_directory("run1", create=False):
        assert isinstance(hippynn.load_model_from_active_directory(), hippynn.GraphModule)
    assert "training_modules" in hippynn.load_checkpoint("experiment_structure.pt", "best_checkpoint.pt", directory="run1")
