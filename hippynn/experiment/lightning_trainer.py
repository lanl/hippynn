import torch

import pytorch_lightning as pl
from .routines import TrainingModules
from ..databases import Database
from .routines import SetupParams, setup_training
from ..graphs import GraphModule
from .controllers import Controller
from .metric_tracker import MetricTracker
from .step_functions import get_step_function, StandardStep
from ..plotting import PlotMaker

class HippynnLightningModule(pl.LightningModule):
    def __init__(self,model: GraphModule,
                 loss: GraphModule,
                 eval_loss: GraphModule,
                 eval_names: list[str],
                 stopping_key: str,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 controller: Controller,
                 metric_tracker: MetricTracker,
                 plot_maker: PlotMaker,
                 inputs:list[str],
                 targets:list[str],
                 *args,**kwargs): # forwards args and kwargs to where?
        super().__init__()

        self.model = model
        self.loss = loss
        self.eval_loss = eval_loss
        self.eval_names = eval_names
        self.stopping_key = stopping_key
        self.controller = controller
        self.metric_tracker = metric_tracker
        self.optimizer = optimizer # does this conflict with PL names?
        self.scheduler = scheduler # does this conflict with PL names?
        self.inputs = inputs
        self.targets = targets
        self.n_inputs = len(self.inputs)
        self.n_targets = len(self.targets)
        self.n_outputs = n_outputs
        self.plot_maker = plot_maker

        self.validation_step_outputs = []

        if not isinstance(step_fn:=get_step_function(optimizer), StandardStep):  # :=
            raise NotImplementedError(f"Optimzers with non-standard steps are not yet supported. {optimizer,step_fn}")
        if args or kwargs:
            raise NotImplementedError("No args or kwargs support yet.")

    @classmethod
    def from_experiment_setup(cls, training_modules: TrainingModules, database: Database, setup_params:SetupParams, **kwargs):
        training_modules, controller, metric_tracker = setup_training(training_modules, setup_params)
        return cls.from_train_setup(training_modules, database, controller, metric_tracker, **kwargs)

    @classmethod
    def from_train_setup(cls,
                         training_modules: TrainingModules,
                         database: Database,
                         controller: Controller,
                         metric_tracker: MetricTracker,
                         callbacks=None,
                         batch_callbacks=None,
                         **kwargs,
                         ):

        model, loss, evaluator = training_modules

        eval_names = evaluator.loss_names
        trainer = cls(
            model = model,
            loss = loss,
            eval_loss = evaluator.loss,
            eval_names = evaluator.loss_names,
            optimizer = controller.optimizer,
            scheduler = controller.scheduler,
            stopping_key = controller.stopping_key,
            controller = controller,
            metric_tracker = metric_tracker,
            plot_maker = evaluator.plot_maker,
            inputs = database.inputs,
            targets = database.targets,
            n_outputs =  evaluator.n_outputs
            **kwargs,
        )


        if callbacks is not None or batch_callbacks is not None:
            return NotImplemented("arbitrary callbacks are not yet supported with pytorch lightning.")

        return trainer, HippynnDataModule(database, controller.batch_size)

    def configure_optimizers(self):

        lr_scheduler_config = {
            "scheduler": self.scheduler,
            "interval": "epoch",  # can be epoch or step
            "frequency": 1,# How many intervals should pass between calls to  `scheduler.step()`.
            "monitor": self.stopping_key, # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "strict": True,
            "name": "learning_rate",
        }

        return dict(optimizer=self.optimizer, lr_scheduler=lr_scheduler_config)


    def training_step(self, batch, batch_idx):

        batch_inputs = batch[:self.n_inputs]
        batch_targets = batch[-self.n_targets:]

        batch_model_outputs = self.model(*batch_inputs)
        batch_train_loss = self.loss(*batch_model_outputs, *batch_targets)[0]

        self.log("train_loss", batch_train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return batch_train_loss

    def validation_step(self, batch, batch_idx):


        batch_inputs = batch[: self.n_inputs]
        batch_targets = batch[-self.n_targets:]

        batch_dict = dict(zip(self.inputs, batch_inputs))
        with torch.autograd.set_grad_enabled(True):
            batch_predictions = self.model(*batch_inputs)

        outputs = (batch_predictions, batch_targets)
        self.validation_step_outputs.append(outputs)
        return batch_predictions

    def on_validation_epoch_end(self,prefix="valid_"):

        all_batch_predictions, all_batch_targets = zip(*self.validation_step_outputs)
        # now 'shape' (n_batch, n_outputs) -> need to transpose.
        all_batch_predictions = [[bpred[i] for bpred in all_batch_predictions] for i in range(self.n_outputs)]
        # now 'shape' (n_batch, n_targets) -> need to transpose.
        all_batch_targets = [[bpred[i] for bpred in all_batch_targets] for i in range(self.n_targets)]

        # now cat each prediction and target across the batch index.
        all_predictions = [torch.cat(x, dim=0) if x[0].shape != () else x[0] for x in all_batch_predictions]
        all_targets = [torch.cat(x, dim=0) for x in all_batch_targets]

        all_losses = [x.item() for x in self.loss(*all_predictions, *all_targets)]
        loss_dict = {name: value for name, value in zip(self.eval_names, all_losses)}
        for k,v in loss_dict.items():
            self.log(prefix + k, v, on_epoch=True, logger=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
        return

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch,batch_idx) # does it work to just run it this way?

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(prefix="test_")



class HippynnDataModule(pl.LightningDataModule):
    def __init__(self, database:Database, batch_size):
        super().__init__()
        self.database = database
        self.batch_size = batch_size

    def train_dataloader(self):
        return self.database.make_generator("train", "train", self.batch_size)

    def val_dataloader(self):
        return self.database.make_generator("valid", "eval", self.batch_size)

    def test_dataloader(self):
        return self.database.make_generator("test", "eval", self.batch_size)