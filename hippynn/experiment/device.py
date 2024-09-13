"""
This file is used to host the function `set_devices`, as it is needed by both
serialization and routines. We may find a better place for this function in the
future.
"""
from typing import Tuple, Union

import torch

from ..graphs import GraphModule
from .evaluator import Evaluator


def set_devices(
    model: GraphModule,
    loss: GraphModule,
    evaluator: Evaluator,
    optimizer: torch.optim.Optimizer,
    device: Union[torch.device, str, Tuple, list],
) -> Tuple:
    """Sets the model, loss, and optimizer to the specified device if necessary.
    Evaluation loss is performed on CPU.

    :param model: current model on CPU
    :param loss: current loss module on CPU
    :param evaluator: evaluator
    :type evaluator: Evaluator
    :param optimizer: optimizer with state dictionary on CPU
    :type optimizer: torch.optim.Optimizer
    :param device: target device for training. If device is a tuple or list, wrap
        the model using torch.nn.DataParallel and use the first specified device
        as the "primary: one.
    :type device: Union[torch.devices, string, tuple, list]
    :return: model, evaluator, optimizer
    :rtype: Tuple
    """
    print("Using device: ", device)
    if isinstance(device, (tuple, list)):
        model = torch.nn.DataParallel(model, device_ids=device)
        print("Using multi GPU compute on devices:", device)
        device = torch.device(device[0])
    device = torch.device(device)  # Will throw a more explicit error if the device specification is invalid

    model.to(device)
    loss.to(device)
    evaluator.loss.cpu()
    evaluator.model_device = device
    evaluator.model = model
    # reload the state dict after model transfer
    optimizer.load_state_dict(optimizer.state_dict())

    return model, evaluator, optimizer
