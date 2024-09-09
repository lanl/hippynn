Pytorch Lightning module
========================


Hippynn incldues support for distributed training using `pytorch-lightning`_.
This can be accessed using the :class:`hippynn.experiment.HippynnLightningModule` class.
The class has two class-methods for creating the lightning module using the same
types of arguments that would be used for an ordinary hippynn experiment.
These are :meth:`hippynn.experiment.HippynnLightningModule.from_experiment_setup`
and :meth:`hippynn.experiment.HippynnLightningModule.from_train_setup`.
Alternatively, you may construct and supply the arguments for the module yourself.

Finally, in additional to the usual pytorch lightning arguments,
the hippynn lightning module saves an additional file, `experiment_structure.pt`,
which needs to be provided as an argument to the
:meth:`hippynn.experiment.HippynnLightningModule.load_from_checkpoint` constructor.


.. _pytorch-lightning: https://github.com/Lightning-AI/pytorch-lightning

