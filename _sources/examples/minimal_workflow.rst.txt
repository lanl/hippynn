Minimal Workflow
================


Here's what you need to do to get going.


First, import the module::

    import hippynn


Next, let's put ourselves in a directory for our first experiment::

    netname = "my_first_hippynn_model"
    dirname = netname
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    os.chdir(dirname)


Now, let's create some nodes. We start with input nodes for the species and positions::

    from hippynn.graphs import inputs, networks, targets, physics, loss

    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")

The ``db_name`` is a key which will be used to find the corresponding information in the database we train or predict on.s

From here, we want to build a neural network. HIPNN has several hyperparameters that should be defined.

These include

* The Z-values of the possible species this network can process.
    (Note, `0` should be included, but entries with a Z value of 0 are treated as "blank atoms")s
* The width of the network, ``n_features``
* The number of sensitivities functions ``n_sensitivities``
* The distance parameters for the start of the sensitivity peaks, the end, and the hard cutoff function.
* The number of interaction blocks
* The number of atom layers in each interaction block

We'll put them in a dictionary::

    network_params = {
        "possible_species": [0,1,6,7,8,16],
        'n_features': 20,
        "n_sensitivities": 20,
        "dist_soft_min": 0.85,
        "dist_soft_max": 5.,
        "dist_hard_max": 7.,
        "n_interaction_layers": 2,
        "n_atom_layers": 3,
    }

We then create a node for our network. The node takes the species and positions, and calculates features.

By passing the node module_kwargs, we are sending the network hyperparameters to the underlying pytorch module::

    network = networks.Hipnn("hipnn_model", (species, positions), module_kwargs = network_params)


From there, we want to define some targets for regression. The ``HEnergyNode`` takes features on individual atoms,
uses them to predict a local quantity, and sums them to create a model energy for the whole system::

    henergy = targets.HEnergyNode("HEnergy",network,db_name="T")

Note again that we have specified the db_name. ``henergy`` is an example of a ``MultiNode`` that has several output attributes. The main output is the energy,
but you can access other children, such as the hierarchicality parameter::

    hierarchicality = henergy.hierarchicality

Having defined these things, let's now define a loss function. The simplest way to do so is with the `of_node` method,
which creates entries for the true and predicted quantities in the database and compares them::

    rmse_energy = loss.MSELoss.of_node(henergy) ** (1 / 2)
    mae_energy = loss.MAELoss.of_node(henergy)

The hierarchicality is an unsupervised quantity, so a loss term there will only depend on the predicted value::

    rbar = loss.Mean.of_node(hierarchicality)

We can combine loss nodes using the regular python syntax for algebra::

    loss_error = (rmse_energy + mae_energy)
    loss = loss_error + rbar

Next we'll define a set of metrics that are tracked between training epochs, as a dictionary.
The keys will define the name used when printing the losses, and the values are the nodes we associate to those names::

    validation_losses = {
        "T-RMSE"      : rmse_energy,
        "T-MAE"       : mae_energy,
        "T-Hier"      : rbar,
        "Error Loss"  : loss_error,
        "Loss"        : loss,
    }

You can put as few or as many metrics as you like.

Having defined a graph structure for the problem, we can tell hippynn to assemble this graph for training::

    training_modules, db_info = hippynn.experiment.assemble_for_training(loss, validation_losses)


The training modules consist of three things:

 1) The model, a ``GraphModule`` that takes the inputs and maps them to predictions
 2) The loss, a ``GraphModule`` for taking predictions and true values and calculating the loss
 3) A model evaluator, which computes all of the validation losses, also from the predictions and true values.

The last thing is the `db_info`, which describes the quantities needed in the database to train to this loss.

It's a simple dictionary containing two lists, the inputs to the model and the targets, or inputs to the loss.

Now we'll load a database::


    database = hippynn.databases.DirectoryDatabase(
        name=<Something>,     # Prefix for arrays in the directory.
        directory=<Somewhere> # Location where the arrays are stored.
        test_size=0.1,  # Fraction or number of samples to test on
        valid_size=0.1, # Fraction or number of samples to validate on
        seed=2001,      # Random seed for spliting data
        ** db_info      # Adds the inputs and targets db_namesnames from the model as things to load
    )

Now that we have a database and a model, we can fit the non-interacting energies using the training set in the database::

    from hippynn.pretraining import hierarchical_energy_initialization
    hierarchical_energy_initialization(henergy,database,trainable_after=False)

We're almost there. We specify the training procedure with ``SetupParams``. We need to have

* The stopping_key used for early stopping
* The batch size for training
* The optimizer to use
* The learning rate
* The maximum number of epochs to train.

Putting it together::

    # Parameters describing the training procedure.
    from hippynn.experiment import SetupParams,setup_and_train

    experiment_params = SetupParams(
        stopping_key="Error Loss",
        batch_size=12,
        optimizer=torch.optim.Adam,
        max_epochs=100,
        learning_rate=0.001,
    )

Now that these are defined, we are good to begin training::

    setup_and_train(training_modules=training_modules,
                    database=database,
                    setup_params=experiment_params,
                    )

When training completes, we can use the model to make predictions.

The simplest form will be to predict everything that the model needs to
compute the loss function. To do this we can use the `from_graph` method
of the predictor, and the `apply_to_db` method to apply it to a database.

The code looks something like this::

    pred = hippynn.graphs.Predictor.from_graph(training_modules.model)
    outputs = pred.apply_to_database(database)

    # The dictionary for the test split
    test_outputs = outputs['test']

    # Get outputs for specific nodes:
    test_hier_predicted = test_outputs[hierarchicality]
    test_energy_predicted = test_outputs[molecule_energy]

Putting this together, we get (more or less) the ``/examples/barebones.py`` file:

.. literalinclude:: /../../examples/barebones.py
