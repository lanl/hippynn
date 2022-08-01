"""
Evaluating network performance over a dataset
"""
import torch
from hippynn.tools import device_fallback, progress_bar


class Evaluator:
    def __init__(self, model, evaluation_loss, evaluation_loss_names, plot_maker=None, plot_every=1, db_info=None):
        """

        :param model: ``GraphModule``. The model to evaluate .
        :param evaluation_loss: ``GraphModule``. The losses to evaluate.
        :param evaluation_loss_names: The names of the losses in the evaluation_loss
        :param plot_maker: An optional plot maker to run.
        :param plot_every: How often to make plots.
        :param db_info:
        """

        self.model = model
        self.loss = evaluation_loss
        self.loss_names = evaluation_loss_names

        self.n_inputs = len(model.input_nodes)
        self.n_outputs = len(model.nodes_to_compute)
        self.n_targets = len(evaluation_loss.input_nodes) - self.n_outputs

        self.plot_maker = plot_maker
        self.db_info = db_info

        self.model_device = device_fallback()

    @property
    def var_list(self):
        return self.db_info["inputs"] + self.db_info["targets"]

    def evaluate(self, generator, eval_type, when=None):
        """
        Construct metrics and activate plotter on a dataloader.
        Note: the model is performed on whatever device it resides on.
        The validation loss is expected to be on the CPU -- this is not too slow,
        and allows for us to evaluate on large datasets whose results would take too much memory
        for the GPU.
        :param generator: the dataset loader
        :param eval_type: str describing dataset, such as 'train', 'valid', 'test'
        :param when: int (epoch #) or str describing when the evaluation is performed
        :return: loss_dict -- dictionary mapping loss names to loss values for this generator
        """

        # Storage for batch prediction values
        prediction_batch_vals = [[] for _ in range(self.n_outputs)]
        target_batch_vals = [[] for _ in range(self.n_targets)]

        # Get the batch prediction values
        for batch in progress_bar(generator, desc=f"Evaluating {eval_type}", unit="batch"):
            batch = [item.to(device=self.model_device) for item in batch]
            batch_inputs = batch[: self.n_inputs]
            batch_targets = batch[-self.n_targets :]
            batch_predictions = self.model(*batch_inputs)
            # Put predictions and targets on CPU
            for storage, value in zip(prediction_batch_vals, batch_predictions):
                storage.append(value.detach().cpu())

            for storage, value in zip(target_batch_vals, batch_targets):
                storage.append(value.detach().cpu())
            del batch_predictions #  To allow freeing memory.

        # Put the batches together
        prediction_all_vals = [torch.cat(x, dim=0) if x[0].shape != () else x[0] for x in prediction_batch_vals]
        target_all_vals = [torch.cat(x, dim=0) for x in target_batch_vals]

        # Evaluate the evaluation losses
        # Note: The `mean` here accounts for device-length vectors returned by scalars computed by
        # a DataParallel module.
        all_losses = [x.mean().item() for x in self.loss(*prediction_all_vals, *target_all_vals)]
        loss_dict = {name: value for name, value in zip(self.loss_names, all_losses)}

        if self.plot_maker:
            self.plot_maker.plot_phase(
                prediction_all_vals=prediction_all_vals, target_all_vals=target_all_vals, when=when, eval_type=eval_type
            )
        return loss_dict
