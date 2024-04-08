Weighted/Masked Loss Functions
==============================

``hippynn`` also supports floating-point-weighted loss functions.
If the weights are all either zero- or one- valued, this is effectively the same as training using a mask.

The sample weights would likely be stored in the database. Let's assume they have the name "W".
We can then create a representation of these weights as input to the problem
as an :class:`~hippynn.graphs.nodes.base.base.InputNode`::

    sample_weights = hippynn.graphs.inputs.InputNode(db_name="W",index_state=hippynn.graphs.IdxType.Molecules)

We use :class:`~hippynn.graphs.indextypes.IdxType.Molecules` in the case of a system-valued target such as energy.
In the case of an atom-valued target such as charge, use :class:`~hippynn.graphs.indextypes.IdxType.MolAtom`.

The weighted loss functions :class:`~hippynn.graphs.nodes.loss.WeightedMSELoss`
and :class:`~hippynn.graphs.nodes.loss.WeightedMAELoss`
can then be constructed as follows::

    weighted_mse_target = hippynn.graphs.loss.WeightedMSELoss.of_node(target, sample_weights)

Mathematically, this will give a loss function

.. math::

    \bar{w} &= \frac{1}{N} \sum_i w_i

    \tilde{w}_i &= w_i/\bar{w}

    L &= \frac{1}{N} \sum_i \tilde{w}_i (\hat{y}_i - y_i)^2

That is, the weights are normalized over the batch, which tends to help stabilize weighted training.

For a fuller example, see the snippet below which mimics the ``/examples/barebones.py`` script::

    # Define a model
    from hippynn.graphs import inputs, networks, targets, physics

    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")

    sample_weights = inputs.InputNode(db_name="W", index_state=hippynn.graphs.IdxType.Molecules)

    network = networks.Hipnn("hipnn_model", (species, positions), module_kwargs=network_params)
    henergy = targets.HEnergyNode("HEnergy", network, db_name="T")

    # define loss quantities
    from hippynn.graphs import loss

    weighted_mse_energy = loss.WeightedMSELoss.of_node(henergy, sample_weights)
    weighted_mae_energy = loss.WeightedMAELoss.of_node(henergy, sample_weights)

    mse_energy = loss.MSELoss.of_node(henergy)
    mae_energy = loss.MAELoss.of_node(henergy)
    rmse_energy = mse_energy ** (1 / 2)

    # Validation losses are what we check on the data between epochs -- we can only train to
    # a single loss, but we can check other metrics too to better understand how the model is training.
    # There will also be plots of these things over time when training completes.
    validation_losses = {
        "RMSE": rmse_energy,
        "MAE": mae_energy,
        'w-MAE': weighted_mae_energy,
        "MSE": mse_energy,
        "w-MSE": weighted_mse_energy,
    }

    training_loss = weighted_mse_energy

Notice that you may wish to report both weighted and unweighted versions of the loss in the validation loss dictionary.