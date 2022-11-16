
Restarting training
===================

You may want to restart training from a previous state, e.g., due to job length
constraints on HPC systems.

By default, hippynn saves checkpoint information for the best model found
so far as measured by validation performance.

The checkpoint contains the training modules, the experiment controller, and
the metrics of the experiment so far. This can be seen by breaking down
:func:`~hippynn.experiment.setup_and_train` into its component steps,
:func:`~hippynn.experiment.setup_training`, and :func:`~hippynn.experiment.routines.train_model`::

    from hippynn.experiment import setup_training, train_model
    training_modules, controller, metrics = setup_training(
        training_modules=training_modules,
        setup_params=experiment_params,
    )
    train_model(
        training_modules,
        database,
        controller,
        metrics,
        callbacks=None,
        batch_callbacks=None,
    )


Simple restart
--------------

To restart training later, you can use the following::

    from hippynn.experiment.serialization import load_checkpoint
    check = load_checkpoint("./experiment_structure.pt", "./best_checkpoint.pt")
    train_model(**check, callbacks=None, batch_callbacks=None)

or to use the default filenames and load from the current directory::

    from hippynn.experiment.serialization import load_checkpoint_from_cwd
    check = load_checkpoint_from_cwd()
    train_model(**check, callbacks=None, batch_callbacks=None)

If all you want to do is use a previously trained model, here is how to load the model only::

    from hippynn.experiment.serialization import load_model_from_cwd
    model = load_model_from_cwd()

The returned ``model`` object will have the original model with the best
parameters loaded. This can then be used with, for example, the :doc:`/examples/predictor`.

Cross-device restart
--------------------

If a model was trained on a device that is no longer accessible (due to change
of configuration or loading on a different computer) you may want to load a checkpoint
to a different device. The standard pytorch argument ``map_device`` is a bit tricky to
handle in this case, as not all tensors in the checkpoint still belong on the device.
If this keyword is specified, ``hippynn`` will attempt to automatically move the correct
tensors to the correct device. To perform cross-device loading, use the ``model_device``
argument to :func:`~hippynn.experiment.serialization.load_checkpoint_from_cwd`
or :func:`~hippynn.experiment.serialization.load_checkpoint`::

     from hippynn.experiment.serialization import load_checkpoint_from_cwd
     check = load_checkpoint_from_cwd(model_device='cuda:0')
     train_model(**check, callbacks=None, batch_callbacks=None)

The string 'auto' can be provided to transfer to the default device.

.. note::
   To avoid cross-device restarts as much as you can, use the
   environment variable ``CUDA_VISIBLE_DEVICES`` before importing ``hippynn``.
   In this case, if, for example, only 1 GPU is used, it will always be
   labeled as 0, no matter physically which device is used.

Advanced Details
----------------

-  The checkpoint file contains the torch RNG state, and restoring a
   checkpoint resets the torch RNG.

-  If your database is not :class:`~hippynn.databases.restarter.Restartable`, you
   will have to explicitly reload it and pass it to ``train_model``, as well.
   If your database is restartable, any pre-processing of the database is not recorded
   in the checkpoint file. Thus any pre-processing steps such as moving the database to
   the GPU need to be performed before activating ``train_model``.
   The dictionary containing the database information is stored as ``training_modules.evaluator.db_info``,
   so you can use this dictionary to easily reload your database.

-  hippynn does not include support for serializing and restarting callback objects; to restart
   a training that involves callbacks, the callbacks will have to be retrieved using user code.


-  It is not a good idea to wholesale transfer tensors in a checkpoint
   off of the CPU using a keyword such as ``map_location=torch.device(0)``.
   This will map all tensors to GPU 0, and breaks the RNG which only supports a CPU
   tensor. Doing so, you will see errors like ``TypeError: RNG state must be a torch.ByteTensor``.
   Moving everything to CPU with ``map_location="cpu"`` always works.
   If ``map_location`` is used, and the value is anything other than ``None`` or ``"cpu"``,
   you are likely to get an error during loading or training.
   The argument will directly be passed to ``torch.load``.

   For more details of this option, check `torch load docs`_. 

   .. _torch load docs: https://pytorch.org/docs/stable/generated/torch.load.html

-  Here are a list of objects and their final device after loading.

   .. list-table::
      :widths: 40 30
      :header-rows: 1

      * - Objects
        - Destinations
      * - ``training_modules.model``
        - ``model_device``
      * - ``training_modules.loss``
        - ``model_device``
      * - ``training_modules.evaluator.loss``
        - CPU
      * - ``controller.optimizer``
        - Some on ``model_device`` and some on CPU,
          depending on details of the implementation.
      * - ``database``
        - CPU
      * - Not mentioned
        - CPU



