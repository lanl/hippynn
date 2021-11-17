Model and Loss Graphs
======================================

The graphs in hippynn are divided into two
conceptual domains, that of the model, and that of the loss.

One reason for this is to cleanly separate what the model
predicts from the true values in the database.
Another reason is to support the separate evaluation of the training loss with all of
the other metrics we may wish to report about the model, in the 'validation loss'.

The distinction can be explicitly made by accessing the ``.pred`` or ``.true`` attributes
of a Node. ``.pred`` corresponds to a predicted computation from the model. ``.true`` is
the corresponding value in the database. This value is found by looking at the ``db_name`` string.
name of the Node.

It is usually sufficient to begin forming a loss graph using the static method ``of_node``::

    from hippynn.graphs import loss
    mae_energy = loss.MAELoss.of_node(molecule_energy)

This is actually equivalent to the more verbose::

    mae_energy = loss.MAELoss(molecule_energy.pred,molecule_energy.true)

It is possible to manually specify computations in the loss graph, rather than the model graph.
However, this is not recommended, because in order to ensure correct statistics, validation losses
are applied to the model outputs of an entire validation subset. For example, the mean of the RMSE
of a set of batches is not equal to the RMSE of all the points in the batches viewed as a single batch.
To deal with this larger size, during validation, the model outputs are detached from autograd and accumulated
on the CPU. Then the validation loss graphs is applied to the entire set of data. This is not typically
expensive so long as the loss graph does not contain any complex operatinos. But one could
desire to do some forms of post-processing in the loss graph. For example, calculating the
per-atom energy in the loss can be accomplished as follows::

    from hippynn.graphs import physics
    pred_per_atom = physics.PerAtom("PeratomPredicted",(molecule_energy.pred,species.true))
    true_per_atom = physics.PerAtom("PeratomTrue",(molecule_energy.true,species.true))
    mae_per_atom = loss.MAELoss(pred_per_atom,true_per_atom)

It is, however, perhaps simpler to pre-calculate the per atom-energy in the database,
and and make it a model output::

    en_per_atom = physics.PerAtom("EnPerAtom",molecule_energy,db_name='EpA')
    mae_per_atom = loss.MAELoss.of_node(en_per_atom)

