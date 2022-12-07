Predictor
=========

The predictor is a simple API for making predictions on an entire database.

Often you'll want to make predictions based on the model. For this,
use :meth:`Predictor.from_graph`. Let's assume you have a ``GraphModule`` called ``model``::

    predictor = hippynn.graphs.Predictor.from_graph(model)

If you just want to make a predictor directly from nodes without having compiled a model for training, it is easy::

    predictor = hippynn.graphs.Predictor(list_of_input_nodes,list_of_output_nodes))

If this fails, it is likely because the inputs do not determine the outputs,
and the error message should say so.

The predictor is callable for making predictions. If all the inputs to the graph
have a db_name, then it is as easy as using them as kwargs.

Let's assume you have a torch tensor z_array for species,
with db_name ``"Z"``, and positions, r_array, with db_name ``"R"``::

    outputs = predictor(Z=z_array,R=r_array)

You can batch the predictions to reduce memory cost with the ``batch_size`` keyword::

    outputs = predictor(Z=z_array,R=r_array,batch_size=128)

``outputs`` is a dictionary, with values of pytorch tensors.
The keys are strings and `Node`s.
To access outputs, you can use the ``db_name`` of the node,
the ``name`` of the node, or the ``Node`` object itself.

If we have a node for molecule energy called ``molecule_energy``
with db_name ``"T"`` and name ``"mol_en"``, this::

    t_predicted_array = outputs['T']

is equivalent to this::

    t_predicted_array = outputs['mol_en']

is equivalent to this::

    t_predicted_array = outputs[molecule_energy]

