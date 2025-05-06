"""
Pytorch Lightning training interface.

This module is somewhat experimental. Using pytorch lightning
successfully in a distributed context may require understanding
and adjusting the various settings related to parallelism, e.g.
multiprocessing context, torch ddp backend, and how they interact
with your HPC environment.

Some features of hippynn experiments may not be implemented yet.
    - The plotmaker is currently not supported.

"""
import warnings
import copy
from pathlib import Path
from typing import Optional

import torch

import pytorch_lightning as pl

from .routines import TrainingModules
from ..databases import Database
from .routines import SetupParams, setup_training
from ..graphs import GraphModule
from .controllers import Controller
from .metric_tracker import MetricTracker
from .step_functions import get_step_function, StandardStep
from ..tools import print_lr
from . import serialization


class HippynnLightningModule(pl.LightningModule):
    """
    A pytorch lightning module for running a hippynn experiment.
    """
    def __init__(
        self,
        model: GraphModule,
        loss: GraphModule,
        eval_loss: GraphModule,
        eval_names: list[str],
        stopping_key: str,
        optimizer_list: list[torch.optim.Optimizer],
        scheduler_list: list[torch.optim.lr_scheduler],
        controller: Controller,
        metric_tracker: MetricTracker,
        inputs: list[str],
        targets: list[str],
        n_outputs: int,
        *args,
        **kwargs,
    ):  # forwards args and kwargs to where?
        super().__init__()

        self.save_hyperparameters(ignore=["loss", "model", "eval_loss", "controller", "optimizer_list", "scheduler_list"])

        self.model = model
        self.loss = loss
        self.eval_loss = eval_loss
        self.eval_names = eval_names
        self.stopping_key = stopping_key
        self.controller = controller
        self.metric_tracker = metric_tracker
        self.optimizer_list = optimizer_list
        self.scheduler_list = scheduler_list
        self.inputs = inputs
        self.targets = targets
        self.n_inputs = len(self.inputs)
        self.n_targets = len(self.targets)
        self.n_outputs = n_outputs

        self.structure_file = None

        self._last_reload_dlene = None  # storage for whether batch size should be changed.

        # Storage for predictions across batches for eval mode.
        self.eval_step_outputs = []
        self.controller.optimizer = None

        for optimizer in self.optimizer_list:
            if not isinstance(step_fn := get_step_function(optimizer), StandardStep):  # :=
                raise NotImplementedError(f"Optimzers with non-standard steps are not yet supported. {optimizer,step_fn}")

        if args or kwargs:
            raise NotImplementedError("Generic args and kwargs not supported.")

    @classmethod
    def from_experiment_setup(cls, training_modules: TrainingModules, database: Optional[Database], setup_params: SetupParams, **kwargs):
        """
        Create a lightning module using the same arguments as for :func:`hippynn.experiment.setup_and_train`.

        :param training_modules:
        :param database:
        :param setup_params:
        :param kwargs:
        :return: lightning_module, database
        """
        training_modules, controller, metric_tracker = setup_training(training_modules, setup_params)
        return cls.from_train_setup(training_modules, database, controller, metric_tracker, **kwargs)

    @classmethod
    def from_train_setup(
        cls,
        training_modules: TrainingModules,
        database: Optional[Database],
        controller: Controller,
        metric_tracker: MetricTracker,
        callbacks=None,
        batch_callbacks=None,
        **kwargs,
    ):
        """
        Create a lightning module from the same arguments as for :func:`hippynn.experiment.train_model`.

        :param training_modules:
        :param database:
        :param controller:
        :param metric_tracker:
        :param callbacks:
        :param batch_callbacks:
        :param kwargs:
        :return: lightning_module, database
        """


        model, loss, evaluator = training_modules

        warnings.warn("PytorchLightning hippynn trainer is still experimental.")

        if evaluator.plot_maker is not None:
            warnings.warn("plot_maker is not currently supported in pytorch lightning. The current plot_maker will be ignored.")

        trainer = cls(
            model=model,
            loss=loss,
            eval_loss=evaluator.loss,
            eval_names=evaluator.loss_names,
            optimizer_list=[controller.optimizer],
            scheduler_list=controller.scheduler_list,
            stopping_key=controller.stopping_key,
            controller=controller,
            metric_tracker=metric_tracker,
            inputs=database.inputs,
            targets=database.targets,
            n_outputs=evaluator.n_outputs,
            **kwargs,
        )

        # pytorch lightning is now in charge of stepping the scheduler.
        controller.scheduler_list = []

        if callbacks is not None or batch_callbacks is not None:
            return NotImplemented("arbitrary callbacks are not yet supported with pytorch lightning.")

        if database is not None:
            database = HippynnDataModule(database, controller.batch_size)

        return trainer, database

    def on_save_checkpoint(self, checkpoint) -> None:
        """

        :param checkpoint:
        :return:
        """

        # Note to future developers:
        # trainer.log_dir property needs to be called on all ranks! This is weird but important;
        # do not move trainer.log_dir inside of a rank zero operation!
        # see https://github.com/Lightning-AI/pytorch-lightning/discussions/8321
        # Thank you to https://github.com/semaphore-egg .
        log_dir = self.trainer.log_dir

        if not self.structure_file:
            # Perform change on all ranks.
            sf = serialization.DEFAULT_STRUCTURE_FNAME
            self.structure_file = sf

        if self.global_rank == 0 and not self.structure_file:
            self.print("creating structure file.")
            structure = dict(
                model=self.model,
                loss=self.loss,
                eval_loss=self.eval_loss,
                controller=self.controller,
                optimizer_list=self.optimizer_list,
                scheduler_list=self.scheduler_list,
            )
            path: Path = Path(log_dir).joinpath(sf)
            self.print("Saving structure file at", path)
            torch.save(obj=structure, f=path)

        checkpoint["controller_state"] = self.controller.state_dict()
        return

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, structure_file=None, hparams_file=None, strict=True, weights_only=False, **kwargs):
        """
        :param checkpoint_path:
        :param map_location:
        :param structure_file:
        :param hparams_file:
        :param strict:
        :param weights_only:
        :param kwargs:

        .. Warning::
        This function uses ``torch.load`` with ``weights_only=False`` by default,
        which can run arbitrary code from the loaded file. Only use it with files
        you trust. You can set ``weights_only=True`` for better security, but
        it may cause loading of the structure file to fail. 

        :return:

        """

        if structure_file is None:
            # Assume checkpoint_path is like <model_name>/version_<n>/checkpoints/<something>.chkpt
            # and that experiment file is stored at <model_name>/version_<n>/experiment_structure.pt
            structure_file = Path(checkpoint_path)
            structure_file = structure_file.parent.parent
            structure_file = structure_file.joinpath(serialization.DEFAULT_STRUCTURE_FNAME)

        structure_args = torch.load(structure_file, weights_only=weights_only)

        return super().load_from_checkpoint(
            checkpoint_path, map_location=map_location, hparams_file=hparams_file, strict=strict, **structure_args, **kwargs
        )

    def on_load_checkpoint(self, checkpoint) -> None:
        """

        :param checkpoint:
        :return:
        """
        cstate = checkpoint.pop("controller_state")
        self.controller.load_state_dict(cstate)
        return

    def configure_optimizers(self):
        """

        :return:
        """

        scheduler_list = []
        for s in self.scheduler_list:
            config = {
                "scheduler": s,
                "interval": "epoch",  # can be epoch or step
                "frequency": 1,  # How many intervals should pass between calls to  `scheduler.step()`.
                "monitor": "valid_" + self.stopping_key,  # Metric to monitor for schedulers like `ReduceLROnPlateau`
                "strict": True,
                "name": "learning_rate",
            }
            scheduler_list.append(config)

        optimizer_list = self.optimizer_list.copy()

        return optimizer_list, scheduler_list

    def on_train_epoch_start(self):
        """

        :return:
        """
        for optimizer in self.optimizer_list:
            print_lr(optimizer, print_=self.print)
        self.print("Batch size:", self.trainer.train_dataloader.batch_size)

    def training_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """

        batch_inputs = batch[: self.n_inputs]
        batch_targets = batch[-self.n_targets:]

        batch_model_outputs = self.model(*batch_inputs)
        batch_train_loss = self.loss(*batch_model_outputs, *batch_targets)[0]

        self.log("train_loss", batch_train_loss)
        return batch_train_loss

    def _eval_step(self, batch, batch_idx):

        batch_inputs = batch[: self.n_inputs]
        batch_targets = batch[-self.n_targets :]

        # It is very, very common to fit to derivatives, e.g. force, in hippynn. Override lightning default.
        with torch.autograd.set_grad_enabled(True):
            batch_predictions = self.model(*batch_inputs)

        batch_predictions = [bp.detach() for bp in batch_predictions]

        outputs = (batch_predictions, batch_targets)
        self.eval_step_outputs.append(outputs)
        return batch_predictions

    def validation_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        return self._eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        return self._eval_step(batch, batch_idx)

    def _eval_epoch_end(self, prefix):

        all_batch_predictions, all_batch_targets = zip(*self.eval_step_outputs)
        # now 'shape' (n_batch, n_outputs) -> need to transpose.
        all_batch_predictions = [[bpred[i] for bpred in all_batch_predictions] for i in range(self.n_outputs)]
        # now 'shape' (n_batch, n_targets) -> need to transpose.
        all_batch_targets = [[bpred[i] for bpred in all_batch_targets] for i in range(self.n_targets)]

        # now cat each prediction and target across the batch index.
        all_predictions = [torch.cat(x, dim=0) if x[0].shape != () else x[0] for x in all_batch_predictions]
        all_targets = [torch.cat(x, dim=0) for x in all_batch_targets]

        all_losses = [x.item() for x in self.eval_loss(*all_predictions, *all_targets)]
        self.eval_step_outputs.clear()  # free memory

        loss_dict = {name: value for name, value in zip(self.eval_names, all_losses)}

        self.log_dict({prefix + k: v for k, v in loss_dict.items()}, sync_dist=True)

        return

    def on_validation_epoch_end(self):
        """

        :return:
        """
        self._eval_epoch_end(prefix="valid_")
        return

    def on_test_epoch_end(self):
        """

        :return:
        """
        self._eval_epoch_end(prefix="test_")
        return

    def _eval_end(self, prefix, when=None) -> None:
        if when is None:
            if self.trainer.sanity_checking:
                when = "Sanity Check"
            else:
                when = self.current_epoch

        # Step 1: get metrics reduced from all ranks.
        # Copied pattern from pytorch_lightning.
        metrics = copy.deepcopy(self.trainer.callback_metrics)

        pre_len = len(prefix)
        loss_dict = {k[pre_len:]: v.item() for k, v in metrics.items() if k.startswith(prefix)}

        loss_dict = {prefix[:-1]: loss_dict}  # strip underscore from prefix and wrap.

        if self.trainer.sanity_checking:
            self.print("Sanity check metric values:")
            self.metric_tracker.evaluation_print(loss_dict, _print=self.print)
            return

        # Step 2: register metrics
        out_ = self.metric_tracker.register_metrics(loss_dict, when=when)
        better_metrics, better_model, stopping_metric = out_
        self.metric_tracker.evaluation_print_better(loss_dict, better_metrics, _print=self.print)

        continue_training = self.controller.push_epoch(self.current_epoch, better_model, stopping_metric, _print=self.print)

        if not continue_training:
            self.print("Controller is terminating training.")
            self.trainer.should_stop = True

        # Step 3: Logic for changing the batch size without always requiring new dataloaders.
        # Step 3a: don't do this when not testing.
        if not self.trainer.training:
            return

        controller_batch_size = self.controller.batch_size
        trainer_batch_size = self.trainer.train_dataloader.batch_size
        if controller_batch_size != trainer_batch_size:
            # Need to trigger a batch size change.
            if self._last_reload_dlene is None:
                # save the original value of this variable to the pl module
                self._last_reload_dlene = self.trainer.reload_dataloaders_every_n_epochs

            # TODO: Make this run even if there isn't an explicit datamodule?
            self.trainer.datamodule.batch_size = controller_batch_size
            # Tell PL lightning to reload the dataloaders now.
            self.trainer.reload_dataloaders_every_n_epochs = 1

        elif self._last_reload_dlene is not None:
            # Restore the last saved value from the pl module.
            self.trainer.reload_dataloaders_every_n_epochs = self._last_reload_dlene
            self._last_reload_dlene = None
        else:
            # Batch sizes match, and there's no variable to restore.
            pass
        return

    def on_validation_end(self):
        """

        :return:
        """
        self._eval_end(prefix="valid_")
        return

    def on_test_end(self):
        """

        :return:
        """
        self._eval_end(prefix="test_", when="test")
        return


class LightingPrintStagesCallback(pl.Callback):
    """
    This callback is for debugging only.
    It prints whenever a callback stage is entered in pytorch lightning.
    """

    for k in dir(pl.Callback):
        if k.startswith("on_"):

            def some_method(self, *args, _k=k, **kwargs):
                all_args = kwargs.copy()
                all_args.update({i: a for i, a in enumerate(args)})
                int_args = {k: v for k, v in all_args.items() if isinstance(v, int)}
                print("Callback stage:", _k, "with integer arguments:", int_args)

            exec(f"{k} = some_method")
            del some_method


class HippynnDataModule(pl.LightningDataModule):
    def __init__(self, database: Database, batch_size):
        super().__init__()
        self.database = database
        self.batch_size = batch_size

    def train_dataloader(self):
        """

        :return:
        """
        return self.database.make_generator("train", "train", self.batch_size)

    def val_dataloader(self):
        """

        :return:
        """
        return self.database.make_generator("valid", "eval", self.batch_size)

    def test_dataloader(self):
        """

        :return:
        """
        return self.database.make_generator("test", "eval", self.batch_size)
