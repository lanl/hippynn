import torch

"""
checkpoint and state generation
"""
from ..databases.restarter import Restartable
from ..tools import device_fallback

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


def __check_mapping_devices(map_location, model_device):
    """Check options for restarting across devices

    Args:
        map_location (Union[int, str, dict, torch.device], optional): device mapping argument for torch.load. Defaults to None.
        model_device (Union[int, str, torch.device], optional): automatically handle device mapping. Defaults to None.

    Raises:
        TypeError: if both map_location and model_device are specified

    Returns:
        tuple: processed map_location and model_device
    """
    # if both are none, no transfer across device happens, directly pass map_location (which is None) to torch.load
    if model_device is not None:
        # if both map_location and model_device are given
        if map_location is not None:
            raise TypeError("Passing map_location explicitly and the model device are incompatible")
        if model_device == "auto":
            model_device = device_fallback()
        map_location = "cpu"
    return map_location, model_device


def __load_saved_tensors(structure_fname, state_fname, **kwargs):
    """Load torch tensors from file.

    Args:
        structure_fname (str): name of the structure file
        state_fname (str): name of the state file

    Returns:
        dict, dict: loaded dictionaries of checkpoint and model parameters
    """

    with open(structure_fname, "rb") as pfile:
        structure = torch.load(pfile, **kwargs)

    with open(state_fname, "rb") as pfile:
        state = torch.load(pfile, **kwargs)
    return structure, state


def load_checkpoint(structure_fname, state_fname, restore_db=True, map_location=None, model_device=None, **kwargs):
    """Load checkpoint file from given filename

    For details on how to use this function, please check the documentations.

    Args:
        structure_fname (str): name of the structure file
        state_fname (str): name of the state file
        restore_db (bool, optional): restore database or not. Defaults to True.
        map_location (Union[int, str, dict, torch.device], optional): device mapping argument for torch.load. Defaults to None.
        model_device (Union[int, str, torch.device], optional): automatically handle device mapping. Defaults to None.

    Returns:
        dict: experiment structure
    """

    map_location, model_device = __check_mapping_devices(map_location, model_device)
    kwargs["map_location"] = map_location
    structure, state = __load_saved_tensors(structure_fname, state_fname, kwargs)

    # transfer stuff back to model_device
    structure = restore_checkpoint(structure, state, restore_db=restore_db)
    # no transfer happens in either case, as the tensors are on the target devices already
    if model_device == "cpu" or map_location != None:
        return structure
    else:
        structure["training_modules"].model.to(model_device)
        structure["training_modules"].loss.to(model_device)
        structure["training_modules"].valuator.model_device = model_device
        structure["training_modules"].valuator.model = structure["training_modules"].model
        return structure


def load_checkpoint_from_cwd(**kwargs):
    """
    See load_checkpoint, but using default filenames.
    :param kwargs:

    :return:
    """
    return load_checkpoint(DEFAULT_STRUCTURE_FNAME, "best_checkpoint.pt", **kwargs)


def load_model_from_cwd(map_location=None, model_device=None, **kwargs):
    """Only load model from current working directory.

    Args:
        map_location (Union[int, str, dict, torch.device], optional): device mapping argument for torch.load. Defaults to None.
        model_device (Union[int, str, torch.device], optional): automatically handle device mapping. Defaults to None.

    Returns:
        torch.nn.Module: model with reloaded parameters
    """

    map_location, model_device = __check_mapping_devices(map_location, model_device)
    kwargs["map_location"] = map_location
    structure, state = __load_saved_tensors("experiment_structure.pt", "best_model.pt", kwargs)

    model = structure["training_modules"].model
    model.load_state_dict(state)

    if model_device == "cpu" or map_location != None:
        return model
    else:
        return model.to(model_device)
