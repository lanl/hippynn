
Restarting training
===================

You may want to restart training from a previous state. (e.g. due to job length
constraints on HPC systems.

By default, hippynn saves checkpoint information for the best model found
so far (on validation data).

The checkpoint contains the training modules, the experiment controller, and
the metrics of the experiment so far. This can be seen by breaking down
:func:`~hippynn.experiment.setup_and_train` into its component steps,
:func:`~hippynn.experiment.setup_training`, and :func:`~hippynn.experiment.routines.train_model`::

    from hippynn.experiment import setup_training, train_modoel
    training_modules,controller,metrics = setup_training(training_modules=training_modules,
                setup_params=experiment_params,
                )
    hippynn.experiment.train_model(training_modules,
                               database,
                               controller,
                               metrics)

To restart training later, you can use the following::

    from hippynn.experiment.serialziation import load_checkpoint
    check = load_checkpoint('./experiment_structure.pt','./best_checkpoint.pkl')
    train_model(**check)

If your database is not Restartable, you wil have to explicitly reload it and pass it to ``train_model``, as well.

The checkpoint file contains and resets the torch RNG state as well.

