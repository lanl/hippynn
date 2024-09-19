"""
Managing optimizer, scheduler, and end of training
"""
import inspect
import warnings

from torch.optim.lr_scheduler import ReduceLROnPlateau

class Controller:
    """
    Class for controlling the training dynamics.

    :param optimizer: pytorch optimizer (or compatible)
    :param scheduler: pytorch scheduler (or compatible)
        Pass None to use no schedulers. May pass a list of schedulers to use them all in sequence.
    :param scheduler_list: list of schedulers (may be empty).
    :param stopping_key: str of name for best metric.
    :param batch_size: batch size for training
    :param eval_batch_size: batch size during evaluation
    :param fraction_train_eval: the random fraction of the training set to use during evaluation.
    :param quiet: If true, suppress printing.

    :param max_epochs: maximum amount of epochs currently allowed based on training so far.

    .. Note::
       If a scheduler defines a ``set_controller`` method, this will be run on the Controller instance.
       This allows a scheduler to modify other aspects of training besides the conventional ones
       found in the optimizer. See ``RaiseBatchSizeOnPlateau`` in this module.

    """

    _state_vars = (
        "boredom",
        "current_epoch",
        "batch_size",
        "eval_batch_size",
        "_max_epochs",
        "fraction_train_eval",
        "stopping_key",
    )

    def __init__(
        self,
        optimizer,
        scheduler,
        batch_size,
        max_epochs,
        stopping_key,
        eval_batch_size=None,
        fraction_train_eval=0.1,
        quiet=False,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.stopping_key = stopping_key
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        if max_epochs is None:
            warnings.warn("Max epochs set to None. Model may train in an infinite loop.")

        self._max_epochs = max_epochs

        self.boredom = 0
        self.current_epoch = 0
        self.quiet = quiet
        self.fraction_train_eval = fraction_train_eval

        try:
            iter(scheduler)
        except TypeError:
            if scheduler is None:
                scheduler = []
            else:
                scheduler = [scheduler]

        for sch in scheduler:
            if hasattr(sch, "set_controller"):
                sch.set_controller(self)

        self.scheduler_list = scheduler

    def state_dict(self):
        state_dict = {k: getattr(self, k) for k in self._state_vars}
        if self.optimizer is not None:
            state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["scheduler"] = [sch.state_dict() for sch in self.scheduler_list]
        return state_dict

    def load_state_dict(self, state_dict):

        for sch, sdict in zip(self.scheduler_list, state_dict["scheduler"]):
            sch.load_state_dict(sdict)

        if self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict["optimizer"])

        for k in self._state_vars:
            setattr(self, k, state_dict[k])

    @property
    def max_epochs(self):
        return self._max_epochs

    def push_epoch(self, epoch, better_model, metric, _print=print):
        self.current_epoch += 1

        if better_model:
            self.boredom = 0
        else:
            self.boredom += 1

        for sch in self.scheduler_list:
            if accepts_metrics(sch):
                sch.step(metrics=metric)
            else:
                sch.step()

        if not self.quiet:
            _print("Epochs since last best:", self.boredom)
            _print("Current max epochs:", self.max_epochs)

        return self.current_epoch < self.max_epochs


class PatienceController(Controller):
    """
    A subclass of Controller that terminates if training has not improved for a given number
    of epochs.

    :ivar patience: How many epochs are allowed without improvement before termination.
    :ivar last_best: The eoch number of the last best epoch encountered.
    """

    _state_vars = Controller._state_vars + ("patience", "last_best")

    def __init__(self, *args, termination_patience, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = termination_patience
        self.last_best = 0

    def push_epoch(self, epoch, better_model, metric, _print=print):
        if better_model:
            if self.boredom > 0 and not self.quiet:
                _print("Patience for training restored.")
            self.boredom = 0
            self.last_best = epoch
        return super().push_epoch(epoch, better_model, metric, _print=_print)

    @property
    def max_epochs(self):
        return min(self.last_best + self.patience + 1, self._max_epochs)


# Developer note: The inheritance here is only so that pytorch lightning
# readily identifies this as a scheduler.
class RaiseBatchSizeOnPlateau(ReduceLROnPlateau):
    """
    Learning rate scheduler compatible with pytorch schedulers.

    Note: The "VERBOSE" Parameter has been deprecated and no longer does anything.

    This roughly implements the scheme outlined in the following paper:

    .. code-block:: none

        "Don't Decay the Learning Rate, Increase the Batch Size"
        Samuel L. Smith et al., 2018.
        arXiv:1711.00489
        Published as a conference paper at ICLR 2018.

    Until max_batch_size has been reached. After that,
    the learning rate is decayed.

    .. Note::
       To use this scheduler, build it, then link it to
       the container which governs the batch size using ``set_controller``.
       The default base hippynn Controller will do this automatically.
    """

    def __init__(
        self,
        optimizer,
        max_batch_size,
        factor=0.5,
        patience=10,
        threshold=0.0001,
        threshold_mode="rel",
        verbose=None, # DEPRECATED
        controller=None,
    ):
        """

        :param optimizer:
        :param max_batch_size:
        :param factor:
        :param patience:
        :param threshold:
        :param threshold_mode:
        :param verbose:
        :param controller:
        """

        if threshold_mode not in ("abs", "rel"):
            raise ValueError("Mode must be 'abs' or 'rel'")

        self.inner = ReduceLROnPlateau(
            optimizer,
            patience=patience,
            factor=factor,
            threshold=threshold,
            threshold_mode=threshold_mode,
        )
        self.controller = controller
        self.max_batch_size = max_batch_size
        self.best_metric = float("inf")
        self.boredom = 0
        self.last_epoch = 0
        warnings.warn("Parameter verbose no longer supported for schedulers. It will be ignored.")

    @property
    def optimizer(self):
        return self.inner.optimizer

    def set_controller(self, box):
        self.controller = box

    def state_dict(self):
        return {
            "boredom": self.boredom,
            "best_metric": self.best_metric,
            "last_epoch": self.last_epoch,
            "inner": self.inner.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.boredom = state_dict["boredom"]
        self.best_metric = state_dict["boredom"]
        self.inner.load_state_dict(state_dict["inner"])
        self.last_epoch = state_dict["last_epoch"]

    def step(self, metrics):
        self.last_epoch += 1

        if self.controller is None:
            warnings.warn(
                "Batch size controller not specified. "
                "Defering to LR Scheduler and batch size will not be controlled."
            )

            return self.inner.step(metrics)

        if self.controller.batch_size >= self.max_batch_size:
            return self.inner.step(metrics)

        if self.inner.threshold_mode == "rel":
            better = self.best_metric * (1 - self.inner.threshold) > metrics
        elif self.inner.treshold_mode == "abs":
            better = self.best_metric - self.inner.threshold > metrics

        if better:
            self.boredom = 0
            self.best_metric = metrics
        else:
            self.boredom += 1

        if self.boredom > self.inner.patience:
            new_batch_size = int(self.controller.batch_size / self.inner.factor)
            new_batch_size = min(new_batch_size, self.max_batch_size)
            self.controller.batch_size = new_batch_size
            self.boredom = 0

            if new_batch_size >= self.max_batch_size:
                self.inner.last_epoch = self.last_epoch - 1

        return


def accepts_metrics(sch):
    """
    Return if a scheduler accepts the validation metric.

    :meta private:
    :param sch:
    :return: bool
    """
    signature = inspect.signature(sch.step)
    return "metrics" in signature.parameters


def is_scheduler_like(thing):
    """
    Test if an object can be used as a scheduler.
    :meta private:
    """
    # Pytorch doesn't have a unified inheritance class for these things.
    # So we call it a scheduler if it has a "step" attribute that itself is callable
    return callable(getattr(thing, "step", None))
