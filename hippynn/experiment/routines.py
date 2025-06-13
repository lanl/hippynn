"""

Routines for setting up and performing training

"""
import sys
import collections
import warnings
from dataclasses import dataclass
import copy
import timeit

from typing import Callable, Union, Optional
import torch

import pickle

from . import serialization
from .metric_tracker import MetricTracker
from .controllers import Controller, is_scheduler_like
from .device import set_devices
from .. import tools
from .assembly import TrainingModules
from .step_functions import get_step_function
from ..databases import Database


from .. import custom_kernels


@dataclass
class SetupParams:
    """
    :param stopping_key:  name of validation metric for stopping
    :param controller: Optional -- Controller object for LR scheduling and ending experiment.
                        If not provided, will be constructed from parameters below.
    :param Device: Where to train the model.
                   Falls back to CUDA if available.
                   Specify a tuple of device numbers to use DataParallel.
    :param max_epochs: Optional -- maximum number of epochs to train.
                    Mandatory if controller not provided.
    :param batch_size:
                    Only used if the controller itself is not specified.
                    Mandatory if controller not provided.
    :param eval_batch_size:
                    Only used if the controller itself is not specified.
    :param scheduler: scheduler passed to the controller
    :param optimizer: Pytorch optimizer or optimizer class. Defaults to Adam.
    :param learning_rate: If an optimizer class is provided, the learning rate is used to construct the optimizer.
    :param fraction_train_eval: What fraction of the training dataset to evaluate in the evaluation phase

    All params after stopping_key, controller, and device are optional and can be built into a controller.

    .. Note::

        Multiple GPUs is an experimental feature that is currently under debugging.
    """

    device: Optional[Union[torch.device, str]] = None
    controller: Optional[Union[Controller, Callable]] = None
    stopping_key: Optional[str] = None
    optimizer: Union[torch.optim.Optimizer, Callable] = torch.optim.Adam
    learning_rate: Optional[float] = None
    scheduler: Union[torch.optim.lr_scheduler._LRScheduler, Callable] = None
    batch_size: Optional[int] = None
    eval_batch_size: Optional[int] = None
    max_epochs: Optional[int] = None
    fraction_train_eval: Optional[float] = 0.1

    _controller_params = (
        "stopping_key",
        "optimizer",
        "learning_rate",
        "scheduler",
        "batch_size",
        "eval_batch_size",
        "max_epochs",
        "fraction_train_eval",
    )

    def __post_init__(self):
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
        if isinstance(self.controller, Controller):
            # Warn if extra arguments are specified, but only if they are not default values.
            for param_str in self._controller_params:
                pval = getattr(self, param_str)
                classpval = getattr(self.__class__, param_str)
                if pval is not None and pval is not classpval:
                    print(
                        "Warning: When controller is specified, argument "
                        f"'{param_str}' should be given to the controller. "
                        f"Value ({getattr(self,param_str)}) ignored.",
                        file=sys.stderr,
                    )
        else:
            # Throw error if not enough arguments specified
            for param_str in self._controller_params:
                if param_str in ("scheduler", "eval_batch_size"):
                    continue
                if getattr(self, param_str) is None:
                    raise ValueError(f"When controller is not specified, argument '{param_str}' must be specified.")


def setup_and_train(
    training_modules: TrainingModules,
    database: Database,
    setup_params: SetupParams,
    store_all_better=False,
    store_best=True,
    store_every=0
):
    """
    :param: training_modules: see :func:`setup_training`
    :param: database: see :func:`train_model`
    :param: setup_params: see :func:`setup_training`
    :param: store_all_better: Save the state dict for each model doing better than a previous one
    :param: store_best: Save a checkpoint for the best model
    :param: store_every: Save a checkpoint for every certain epochs
    :return: See :func:`train_model`

    Shortcut for setup_training followed by train_model.

    .. Note::
        The training loop will capture KeyboardInterrupt exceptions to abort the experiment early.
        If you would like to gracefully kill training programmatically, see :func:`train_model` with callbacks argument.
    .. Note::
        Saves files in the current running directory; recommend you switch
        to a fresh directory with a descriptive name for your experiment.

    """
    # Set up objects for training.

    training_modules, controller, metric_tracker = setup_training(
        training_modules=training_modules, setup_params=setup_params
    )

    # Actually do the training
    return train_model(
        training_modules=training_modules,
        database=database,
        controller=controller,
        metric_tracker=metric_tracker,
        callbacks=None,
        batch_callbacks=None,
        store_all_better=store_all_better,
        store_best=store_best,
        store_every=store_every
    )


def setup_training(
    training_modules: TrainingModules,
    setup_params: SetupParams,
):
    """
    Prepares training_modules for training with experiment_params.

    :param: training_modules: Tuple of model, training loss, and evaluation losses
            (Can be built from graph using `graphs.assemble_training_modules`)
    :param: setup_params: parameters controlling how training is performed
            (See :class:`SetupParams`)

    Roughly:

    * sets devices for training modules
    * if no controller given:

       * instantiates and links optimizer to the learnable params on the model
       * instantiates and links scheduler to optimizer
       * builds a default controller with setup params
    * creates a MetricTracker for storing the training metrics

    :return: (optimizer,evaluator,controller,metrics,callbacks)
    """
    if isinstance(setup_params, dict):
        setup_params = SetupParams(**setup_params)

    model, loss, evaluator = training_modules

    controller = setup_params.controller
    if not isinstance(controller, Controller):
        print("Generating controller using values from setup params.")

        optimizer = setup_params.optimizer
        if not isinstance(optimizer, torch.optim.Optimizer) and callable(optimizer):
            print("Generating Optimizer from setup params, defaulting to parameters in model that require grad.")
            params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = optimizer(params, lr=setup_params.learning_rate)

        scheduler = setup_params.scheduler
        if not is_scheduler_like(scheduler) and callable(scheduler):
            print("Generating Scheduler from setup params")
            scheduler = scheduler(optimizer)

        controller_cls = Controller if controller is None else controller

        controller = controller_cls(
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=setup_params.batch_size,
            eval_batch_size=setup_params.eval_batch_size,
            max_epochs=setup_params.max_epochs,
            stopping_key=setup_params.stopping_key,
            fraction_train_eval=setup_params.fraction_train_eval,
        )

    optimizer = controller.optimizer
    model, evaluator, optimizer = serialization.set_devices(
        model, loss, evaluator, optimizer, setup_params.device or tools.device_fallback()
    )

    metrics = MetricTracker(evaluator.loss_names,
                            stopping_key=controller.stopping_key)
    return training_modules, controller, metrics


def train_model(
    training_modules,
    database,
    controller,
    metric_tracker,
    callbacks,
    batch_callbacks,
    store_all_better=False,
    store_best=True,
    store_every=0,
    store_structure_file=True,
    store_metrics=True,
    quiet=False,
):
    """
    Performs training loop, allows keyboard interrupt. When done,
    reinstate the best model, make plots and metrics over time, and test the model.

    :param training_modules: tuple-like of model, loss, and evaluator
    :param database: Database
    :param controller: Controller
    :param metric_tracker: MetricTracker for storing model performance
    :param callbacks: callbacks to perform after every epoch.
    :param batch_callbacks: callbacks to perform after every batch
    :param store_best: Save a checkpoint for the best model
    :param store_all_better: Save the state dict for each model doing better than a previous one
    :param store_every: Save a checkpoint for every certain epochs
    :param store_structure_file: Save the structure file for this experiment
    :param store_metrics: Save the metric tracker for this experiment.
    :param quiet: If True, disable printing during training (still prints testing results).

    :return: metric_tracker

    .. Note::
        callbacks take the form of an iterable of callables
        and will be called with cb(epoch,new_best)

        * epoch indicates the epoch number
        * new_best indicates if the model is a new best model

    .. Note::
        batch_callbacks take the form of an  iterable of callables
        and will each be called with cb(batch_inputs, batch_model_outputs, batch_targets)

    .. Note::
        You may want to make your callbacks store other state, if so, an easy way is to make them a callable object.

    .. Note::
        callback state is not managed by ``hippynn``. If your wish to save or load callback state, you will
        have to manage that manually (possibly with a callback itself).

    """

    model, loss, evaluator = training_modules

    print("Beginning training.")
    if not quiet:
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            print_model = model.module
            print("Distributed training with ", type(model))
        else:
            print_model = model
        try:
            print("Model:")
            print_model.print_structure()
        except Exception as eee:
            print(f"Model structure couldn't be printed. (Got {type(eee)}). Skipping.")
        print("Model Params:")
        tools.param_print(print_model)

    metric_tracker.quiet = quiet

    # Ensure database inputs and targets are available and ordered correctly.
    db_input_set = set(database.inputs)
    db_target_set = set(database.targets)
    evaluator_input_set = set(evaluator.db_info['inputs'])
    evaluator_target_set = set(evaluator.db_info['targets'])

    if db_input_set != evaluator_input_set:        
        raise ValueError("Evaluator and database have incompatible input sets: "
                        f"{db_input_set} vs {evaluator_input_set}")
    if db_target_set != evaluator_target_set:
        raise ValueError("Evaluator and database have incompatible target sets: "
                        f"{db_target_set} vs {evaluator_target_set}")
    database.inputs = evaluator.db_info['inputs']
    database.targets = evaluator.db_info['targets']

    if store_structure_file:
        serialization.create_structure_file(training_modules, database, controller)

    try:
        training_loop(
            training_modules=training_modules,
            database=database,
            controller=controller,
            metric_tracker=metric_tracker,
            callbacks=callbacks,
            batch_callbacks=batch_callbacks,
            store_best=store_best,
            store_all_better=store_all_better,
            store_every=store_every,
            quiet=quiet,
        )

    except KeyboardInterrupt:
        print("******* TRAINING INTERRUPTED *******")
        print("Finishing up...")
    print("Training phase ended.")

    torch.save(metric_tracker, "training_metrics.pt")

    best_model = metric_tracker.best_model
    if best_model:
        print("Reverting to best model found.")
        model.load_state_dict(best_model)

    print("Making plots over training time...")
    metric_tracker.plot_over_time()
    print("Testing model...")
    torch.cuda.empty_cache()
    test_model(
        database,
        evaluator,
        when="FinalTraining",
        batch_size=controller.eval_batch_size,
        metric_tracker=metric_tracker,
    )

    if store_metrics:
        with open("training_metrics.pkl", "wb") as pfile:
            pickle.dump(metric_tracker, pfile)
    print("Training complete.")
    return metric_tracker


def test_model(database, evaluator, batch_size, when, metric_tracker=None):
    """
    Tests the model on the database according to the model_evaluator metrics.
    If a plot_maker is attached to the model evaluator, it will make plots.
    The plots will go in a sub-folder specified by `when` the testing is taking place.
    The results are then printed.

    :param database: The database test the model on.
    :param evaluator: The evaluator containing model and evaluation losses to measure.
    :param when: A string to specify what plots are currently to be used.
    :param metric_tracker: (Optional) metric tracker to save metrics on. If not provided,
        a blank one will be constructed.

    :return: metric tracker
    """
    evaluator.model.eval()

    if metric_tracker is None:
        metric_tracker = MetricTracker(evaluator.loss_names, stopping_key=None)

    # Determine splits which are complete and can be evaluated:
    evaluatable_splits = []
    required_variables = set(database.inputs + database.targets)
    for sname, split in database.splits.items():
        if all(k in split for k in required_variables):
            evaluatable_splits.append(sname)
        else:
            missing_arrays = set(k for k in required_variables if k not in split)
            warnings.warn(f"Database contains split '{sname}' which"
                          f" cannot be evaluated because it does not contain the"
                          f" required quantities: {missing_arrays}")

    # A little dance to make sure train, valid, test always come first, when present.
    basic_splits = ["train", "valid", "test"]
    basic_splits = [s for s in basic_splits if s in evaluatable_splits]
    splits = basic_splits + [s for s in evaluatable_splits if s not in basic_splits]

    evaluation_data = collections.OrderedDict(
        (
            (key, database.make_generator(key, "eval", batch_size))
            for key in splits
        )
    )
    evaluation_metrics = {k: evaluator.evaluate(gen, eval_type=k, when=when) for k, gen in evaluation_data.items()}
    metric_tracker.register_metrics(evaluation_metrics, when=when)
    metric_tracker.evaluation_print(evaluation_metrics, quiet=False)
    return metric_tracker


def training_loop(
    training_modules: TrainingModules,
    database,
    controller: Controller,
    metric_tracker: MetricTracker,
    callbacks,
    batch_callbacks,
    store_all_better,
    store_best,
    store_every,
    quiet,
):
    """
    Performs a high-level training loop.

    :param training_modules:  training modules from ``assemble_modules``
    :param database: database to train to
    :param controller: controller for early stopping and/or learning rate decay
    :param metric_tracker: the metric tracker
    :param callbacks: list of callbacks for each epoch
    :param batch_callbacks: list of callbacks for each batch
    :param store_best: Save a checkpoint for the best model
    :param store_all_better: Save the state dict for each model doing better than a previous one
    :param store_every: Save a checkpoint for every certain epochs
    :param quiet: whether to print information. Setting quiet to true won't prevent progress bars.

    :return: metrics -- the state of the experiment after training

    .. Note::
        Saves files in the current running directory; recommend switching
        to a fresh directory with a descriptive name for each experiment.

    Rough structure.

    Loop over Epochs, performing:

    - Loop over batches

        * Make predictions
        * Calculate Loss, perform backwards
        * Optimizer Step
        * Batch Callbacks
    - Perform validation and print results
    - Controller/Scheduler Step
    - Epoch callbacks
    - If new best, save the model state_dict and a checkpoint

    """

    if quiet:
        qprint = lambda *args, **kwargs: None
    else:
        qprint = print

    model, loss, evaluator = training_modules

    ### Trigger better error message if inputs or targets is not set!
    database.var_list
    # (If we remove this line, we'll still get an error but it will be more obscure.)
    ###
    n_inputs = len(database.inputs)
    n_targets = len(database.targets)

    qprint("At least {} epochs will be run".format(controller.max_epochs))

    epoch = metric_tracker.current_epoch
    device = evaluator.model_device
    step_function = get_step_function(controller.optimizer)
    optimizer = controller.optimizer

    continue_training = True  # Assume that nobody ran this function without wanting at least 1 epoch.

    while continue_training:

        qprint("_" * 50)
        qprint("Epoch {}:".format(epoch))
        tools.print_lr(optimizer)
        qprint("Batch Size:", controller.batch_size)

        qprint(flush=True, end="")

        model.train()
        epoch_run_time = timeit.default_timer()
        train_generator = database.make_generator("train", "train", batch_size=controller.batch_size)

        for batch in tools.progress_bar(train_generator, desc="Training Batches", unit="batch"):

            batch = [item.to(device=device, non_blocking=True) for item in batch]
            batch_inputs = batch[:n_inputs]
            batch_targets = batch[-n_targets:]
            batch_targets = [x.requires_grad_(False) for x in batch_targets]

            batch_model_outputs = step_function(optimizer, model, loss, batch_inputs, batch_targets)

            if batch_callbacks:
                for cb in batch_callbacks:
                    cb(batch_inputs, batch_model_outputs, batch_targets)
            # Allow garbage collection of computed values.
            del batch_model_outputs

        elapsed_epoch_run_time = timeit.default_timer() - epoch_run_time
        qprint("Training time: ", round(elapsed_epoch_run_time, 2), "s")
        qprint("Validating...", flush=True)

        model.eval()

        evaluation_data = collections.OrderedDict(
            (
                (
                    "train",
                    database.make_generator(
                        "train", "eval", batch_size=controller.eval_batch_size, subsample=controller.fraction_train_eval
                    ),
                ),
                ("valid", database.make_generator("valid", "eval", controller.eval_batch_size)),
            )
        )
        evaluation_metrics = {k: evaluator.evaluate(gen, eval_type=k, when=epoch) for k, gen in evaluation_data.items()}

        better_metrics, better_model, stopping_metric = metric_tracker.register_metrics(evaluation_metrics, when=epoch)
        metric_tracker.evaluation_print_better(evaluation_metrics, better_metrics)

        # Check Learning Rate Schedule
        continue_training = controller.push_epoch(epoch, better_model, stopping_metric)

        elapsed_epoch_run_time = timeit.default_timer() - epoch_run_time
        metric_tracker.epoch_times.append(elapsed_epoch_run_time)
        qprint("Total epoch time: ", round(elapsed_epoch_run_time, 2), "s")

        if callbacks:
            for cb in callbacks:
                cb(epoch, better_model)

        if better_model:
            qprint("**** NEW BEST MODEL - Saving! ****")
            best_model = copy.deepcopy(model.state_dict())
            metric_tracker.best_model = best_model

            if store_all_better:
                # Save a copy of every network doing better
                # Note: epoch has already been incremented, so decrement in saving file.
                with open(f"better_model_epoch_{epoch}.pt", "wb") as pfile:
                    torch.save(best_model, pfile)

            if store_best:
                # Overwrite the "best model so far"
                with open("best_model.pt", "wb") as pfile:
                    torch.save(best_model, pfile)

                state = serialization.create_state(model, controller, metric_tracker)

                # Write the checkpoint
                with open("best_checkpoint.pt", "wb") as pfile:
                    torch.save(state, pfile)
            
        if store_every and epoch != 0 and (epoch % store_every) == 0:
            # Save a copy every "store_every" epoch
            with open(f"model_epoch_{epoch}.pt", "wb") as pfile:
                torch.save(model.state_dict(), pfile)
            
            state = serialization.create_state(model, controller, metric_tracker)

            # Write the checkpoint
            with open(f"checkpoint_epoch_{epoch}.pt", "wb") as pfile:
                torch.save(state, pfile)

        epoch += 1

    return metric_tracker
