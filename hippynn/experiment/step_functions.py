"""
This file implements various stepping protocols used by different optimizer APIs.

In particular:
    - The "standard" step function which only requires that backwards has been called.
    - The "closure" step function for when line search is required (currently only active on LBFGS)
    - The "two step" style of Sharpness Aware Minimization algorithms

The main output function here is `get_step_function(optimizer)-> callable`.

The various step functions are provided as classes that act with staticmethods.
This is to provide for the possibility of extension, for example, to schemes with
stepping schemes that require additional state, or for the possibility to specifiy
the step function explicitly within the controller.
"""
from torch.optim import Optimizer, LBFGS


def standard_step_fn(optimizer, model, loss, batch_inputs, batch_targets):
    optimizer.zero_grad(set_to_none=True)
    batch_model_outputs = model(*batch_inputs)

    # The extra .mean call here deals with an edge case for multi-GPU DataParallel with scalar outputs
    batch_train_loss = loss(*batch_model_outputs, *batch_targets)[0].mean()

    batch_train_loss.backward()
    optimizer.step()
    return batch_model_outputs


def twostep_step_fn(optimizer, model, loss, batch_inputs, batch_targets):
    # Step function for SAM algorithm.
    optimizer.zero_grad(set_to_none=True)

    batch_model_outputs = model(*batch_inputs)
    batch_train_loss = loss(*batch_model_outputs, *batch_targets)[0].mean()
    batch_train_loss.backward()
    optimizer.first_step(zero_grad=True)

    batch_model_outputs_2 = model(*batch_inputs)
    loss(*batch_model_outputs_2, *batch_targets)[0].mean().backward()
    optimizer.second_step(zero_grad=True)
    return batch_model_outputs


def closure_step_fn(optimizer, model, loss, batch_inputs, batch_targets):
    return_outputs = None

    def closure():
        nonlocal return_outputs
        optimizer.zero_grad(set_to_none=True)
        batch_model_outputs = model(*batch_inputs)
        if return_outputs is None:
            return_outputs = batch_model_outputs
        batch_train_loss = loss(*batch_model_outputs, *batch_targets)[0].mean()
        batch_train_loss.backward()
        return batch_train_loss

    optimizer.step(closure)
    return return_outputs


# Note: The staticmethod version here can be re-written using class parameters
# and __init_subclass, but will they always be staticmethods?
class StepFn:
    step = NotImplemented

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)


class StandardStep(StepFn):
    step = staticmethod(standard_step_fn)


class TwoStep(StepFn):
    step = staticmethod(twostep_step_fn)


class ClosureStep(StepFn):
    step = staticmethod(closure_step_fn)


def get_step_function(optimizer: Optimizer) -> callable:
    if type(optimizer).__name__ == "SAM":
        return TwoStep()
    if isinstance(optimizer, (LBFGS,)):
        return ClosureStep()
    else:
        return StandardStep()
