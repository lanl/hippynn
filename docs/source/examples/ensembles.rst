Ensembling Models
#################


Using the :func:`~hippynn.graphs.make_ensemble` function makes it easy to combine models.

By default, ensembling is based on the db_name for the nodes in each input graph.
Nodes which have the same name will be assigned an ensemble node which combines
the different versions of that quantity, and additionally calculates the
mean and standard deviation.

It is easy to make an ensemble from a glob string or a list of directories where
the models are saved::

    from hippynn.graphs import make_ensemble
    model_form = '../../collected_models/quad0_b512_p5_GPU*'
    ensemble_graph, ensemble_info = make_ensemble(model_form)

The ensemble graph takes the inputs which are required for all of the models in the ensemble.
The ``ensemble_info`` object provides the counts for the inputs and targets of the ensemble
and the counts of those corresponding quantities across the ensemble members.

A typical use case would be to then build a Predictor or ASE Calculator from the ensemble.
See :file:`~examples/ensembling_models.py` for a detailed example.

