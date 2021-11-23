import torch

"""
checkpoint and state generation
"""
from ..databases.restarter import Restartable

DEFAULT_STRUCTURE_FNAME = "experiment_structure.pt"


def create_state(model, controller, metric_tracker):
    """
    Create an experiment state dictionary.

    :param model:
    :param controller:
    :param metric_tracker:

    :return: dictionary containing experiment state.
    """
    return {
        "model": model.state_dict(),
        "controller": controller.state_dict(),
        "metric_tracker": metric_tracker,
        "torch_rng_state": torch.random.get_rng_state(),
    }


def create_structure_file(training_modules, database, controller, fname=DEFAULT_STRUCTURE_FNAME):
    """
    Save an experiment structure. (i.e. full model, not just state_dict).

    :param training_modules:
    :param database:
    :param controller:
    :param fname:

    :return: Nothing
    """
    structure = {
        "training_modules": training_modules,
        "controller": controller,
    }
    if isinstance(database, Restartable):
        structure["database"] = database.restarter

    with open(fname, "wb") as pfile:
        torch.save(structure, pfile)


def restore_checkpoint(structure, state, restore_db=True):
    """

    :param structure: experiment structure object
    :param state: experiment state object
    :param restore_db: Attempt to restore database (true/false)

    :return: experiment structure
    """

    structure["training_modules"][0].load_state_dict(state["model"])
    structure["controller"].load_state_dict(state["controller"])

    if "database" in structure and restore_db:
        structure["database"] = structure["database"].attempt_reload()

    structure["metric_tracker"] = state["metric_tracker"]
    torch.random.set_rng_state(state["torch_rng_state"])

    return structure


def load_checkpoint(structure_fname, state_fname, restore_db=True, **kwargs):
    """
    Load a checkpoint from filenames. kwargs are passed to torch

    :param structure_fname:
    :param state_fname:
    :param restore_db:
    :param kwargs: passed to torch.load, i.e. use `map_location` to load the model
     on a specific device

    :return:
    """

    with open(structure_fname, "rb") as pfile:
        structure = torch.load(pfile, **kwargs)

    with open(state_fname, "rb") as pfile:
        state = torch.load(pfile, **kwargs)

    return restore_checkpoint(structure, state, restore_db=restore_db)


def load_checkpoint_from_cwd(**kwargs):
    """
    See load_checkpoint, but using default filenames.
    :param kwargs:

    :return:
    """
    return load_checkpoint(DEFAULT_STRUCTURE_FNAME, "best_checkpoint.pt", **kwargs)


def load_model_from_cwd(**kwargs):
    """
    Loads structure and best model params from cwd, returns model only.
    :param kwargs: passed to torch.load

    :return:
    """

    with open("experiment_structure.pt", "rb") as pfile:
        structure = torch.load(pfile, **kwargs)

    with open("best_model.pkl", "rb") as pfile:
        state = torch.load(pfile, **kwargs)

    model = structure["training_modules"].model
    model.load_state_dict(state)

    return model
