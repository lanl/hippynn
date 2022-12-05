"""
checkpoint and state generation
"""

from typing import Tuple, Union

import torch

from ..databases import Database
from ..databases.restarter import Restartable
from ..graphs import GraphModule
from ..tools import device_fallback
from .assembly import TrainingModules
from .controllers import PatienceController
from .device import set_devices
from .metric_tracker import MetricTracker

DEFAULT_STRUCTURE_FNAME = "experiment_structure.pt"


def create_state(
    model: GraphModule,
    controller: PatienceController,
    metric_tracker: MetricTracker,
) -> dict:
    """Create an experiment state dictionary.

    :param model: current model
    :param controller: patience controller
    :param metric_tracker: current metrics
    :return: dictionary containing experiment state.
    :rtype: dict
    """
    return {
        "model": model.state_dict(),
        "controller": controller.state_dict(),
        "metric_tracker": metric_tracker,
        "torch_rng_state": torch.random.get_rng_state(),
    }


def create_structure_file(
    training_modules: TrainingModules,
    database: Database,
    controller: PatienceController,
    fname=DEFAULT_STRUCTURE_FNAME,
) -> None:
    """
    Save an experiment structure. (i.e. full model, not just state_dict).

    :param training_modules: contains model, controller, and loss
    :param database: database for training
    :param controller: patience controller
    :param fname: filename to save the checkpoint

    :return: None
    """
    structure = {
        "training_modules": training_modules,
        "controller": controller,
    }
    if isinstance(database, Restartable):
        structure["database"] = database.restarter

    with open(fname, "wb") as pfile:
        torch.save(structure, pfile)


def restore_checkpoint(structure: dict, state: dict, restore_db=True) -> dict:
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


def check_mapping_devices(map_location, model_device):
    """
    Check options for restarting across devices.

    :param map_location: device mapping argument for torch.load.
    :param model_device: automatically handle device mapping.
    :raises TypeError: if both map_location and model_device are specified
    :return: processed map_location and model_device
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


def load_saved_tensors(structure_fname: str, state_fname: str, **kwargs) -> Tuple[dict, dict]:
    """Load torch tensors from file.

    :param structure_fname: name of the structure file
    :param state_fname: name of the state file
    :return: loaded dictionaries of checkpoint and model parameters
    """

    with open(structure_fname, "rb") as pfile:
        structure = torch.load(pfile, **kwargs)

    with open(state_fname, "rb") as pfile:
        state = torch.load(pfile, **kwargs)
    return structure, state


def load_checkpoint(
    structure_fname: str, state_fname: str, restore_db=True, map_location=None, model_device=None, **kwargs
) -> dict:
    """
    Load checkpoint file from given filename.

    For details more information on to use this function, see :doc:`/examples/restarting`.

    :param structure_fname: name of the structure file
    :param state_fname: name of the state file
    :param restore_db: restore database or not, defaults to True
    :param map_location: device mapping argument for ``torch.load``, defaults to None
    :param model_device: automatically handle device mapping. Defaults to None, defaults to None
    :return: experiment structure
    """

    # we need keep the original map_location value for the if
    mapped, model_device = check_mapping_devices(map_location, model_device)
    kwargs["map_location"] = mapped
    structure, state = load_saved_tensors(structure_fname, state_fname, **kwargs)

    # transfer stuff back to model_device
    structure = restore_checkpoint(structure, state, restore_db=restore_db)
    # no transfer happens in either case, as the tensors are on the target devices already
    if model_device == "cpu" or map_location != None:
        evaluator = structure["training_modules"].evaluator
        # model_device can be None, so we have to determine the device for a tensor
        evaluator.model_device = next(evaluator.model.parameters()).device
    # if map_location is not set and model_device is set
    elif model_device != None:
        training_modules = structure["training_modules"]
        optimizer = structure["controller"].optimizer
        model, loss, evaluator = training_modules
        model, evaluator, optimizer = set_devices(model, loss, evaluator, optimizer, model_device)
    # if neither map_location nor model_device is set, directly return
    return structure


def load_checkpoint_from_cwd(map_location=None, model_device=None, **kwargs) -> dict:
    """
    Same as ``load_checkpoint``, but using default filenames.

    :param map_location: device mapping argument for ``torch.load``, defaults to None
    :type map_location: Union[str, dict, torch.device, Callable], optional
    :param model_device: automatically handle device mapping. Defaults to None, defaults to None
    :type model_device: Union[int, str, torch.device], optional
    :return: experiment structure
    :rtype: dict
    """
    return load_checkpoint(
        DEFAULT_STRUCTURE_FNAME, "best_checkpoint.pt", map_location=map_location, model_device=model_device, **kwargs
    )


def load_model_from_cwd(map_location=None, model_device=None, **kwargs) -> GraphModule:
    """
    Only load model from current working directory.

    :param map_location: device mapping argument for ``torch.load``, defaults to None
    :type map_location: Union[str, dict, torch.device, Callable], optional
    :param model_device: automatically handle device mapping. Defaults to None, defaults to None
    :type model_device: Union[int, str, torch.device], optional
    :return: model with reloaded parameters
    :rtype: GraphModule
    """
    mapped, model_device = check_mapping_devices(map_location, model_device)
    kwargs["map_location"] = mapped
    structure, state = load_saved_tensors("experiment_structure.pt", "best_model.pt", **kwargs)

    model = structure["training_modules"].model
    model.load_state_dict(state)
    if map_location == None and model_device != None and model_device != "cpu":
        model = model.to(model_device)
    return model
