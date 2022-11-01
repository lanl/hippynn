
Restarting training
===================

You may want to restart training from a previous state, e.g., due to job length
constraints on HPC systems.

By default, hippynn saves checkpoint information for the best model found
so far (on validation data).

The checkpoint contains the training modules, the experiment controller, and
the metrics of the experiment so far. This can be seen by breaking down
:func:`~hippynn.experiment.setup_and_train` into its component steps,
:func:`~hippynn.experiment.setup_training`, and :func:`~hippynn.experiment.routines.train_model`::

    from hippynn.experiment import setup_training, train_model
    training_modules, controller, metrics = setup_training(
        training_modules=training_modules,
        setup_params=experiment_params,
    )
    hippynn.experiment.train_model(
        training_modules,
        database,
        controller,
        metrics,
        callbacks,
        batch_callbacks,
    )

Note that the database is always placed on CPU after restarting. To transfer it
to GPU, you have to do it explicitly through ``database.send_do_device(desired_device)``.

Also note that it is not possible in general to save callbacks, so when using
this function, you need reconstruct your previous callbacks manually or set the
two parameters to ``None``.

There are two types restart,

1. :ref:`A simple restart <simple>`,
2. :ref:`A cross-device restart <cross>`, where tensors will be mapped to a
   different device.

.. _simple:

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

If your database is not :class:`~hippynn.databases.restarter.Restartable`, you
will have to explicitly reload it and pass it to ``train_model``, as well. The
dictionary containing the database information is stored as ``training_modules.evaluator.db_info``,
so you can use this dictionary to easily reload your database.

The checkpoint file contains and resets the torch RNG state.

Alternatively, it is possible to load the model only::

    from hippynn.experiment.serialization import load_model_from_cwd
    model = load_model_from_cwd()

The returned ``model`` object will have the original model with the best
parameters loaded. Of course, to actually use the model, you need create other
object manually.

.. _cross:

Cross-device restart
--------------------

.. role:: red

**A quick tip**: to avoid cross-device restarts as much as you can, use the
environment variable ``CUDA_VISIBLE_DEVICES`` from shell or set it before
importing hippynn (:red:`Important !`), instead of setting devices inside your
script. In this case, if, for example, only 1 GPU is used, it will always be
labeled as 0, no matter physically which device is used.

#######

It is a lot trickier to reload a model or checkpoint across devices. At this
moment, we provide the following possibilities.

#. You explicitly know the original and new devices used. For example, to 
   transfer all tensors that are on GPU 1 to GPU 0::
   
    from hippynn.experiment.serialization import load_checkpoint_from_cwd
    check = load_checkpoint_from_cwd(map_location={"cuda:1": "cuda:0"})
    train_model(**check)

   The dictionary is a explicitly mapping for the old device (key) to the new
   device (value). So if a tensor that is not on the old device will not be
   transferred. For example, in the above example, tensors on CPU will stay.

   Note that:

   #. As aforementioned, the database (if restarted) will be loaded to CPU. An
      manual transfer is still needed.
   #. If ``map_location`` is used and the value is anything other than ``None``,
      we will not handle any exception. The argument will directly be passed to
      ``torch.load``. Use this only if you are 100% about the devices.

   For more details of this option, check `torch load docs`_. 

   .. _torch load docs: https://pytorch.org/docs/stable/generated/torch.load.html

#. Leave the problem to us via the ``model_device`` option. If this option is
   given, all tensors will first be transferred to CPU and then transferred to
   ``model_device`` if necessary. Note only some tensors will be transferred to
   GPU if a GPU is available.

   #. ``model_device="auto"`` :func:`~hippynn.tools.device_fallback` will be
      used to automatically select the best device. If there is GPU, GPU will be
      selected. If there are multiple GPUs, GPU 0 will be chosen. Otherwise, we
      will use CPU.

   #. ``model_device="cpu"`` or ``model_device=0`` or ``model_device="cuda:1"``
      or ``model_device=torch.device(2)`` Given device will be used as to load
      tensors. Make sure the target device is available.

   :func:`~hippynn.experiment.serialization.load_model_from_cwd` works exactly
   the same.

   Here are a list of objects and their final device after loading.

   .. list-table::
      :widths: 40 30
      :header-rows: 1

      * - Objects
        - Destinations
      * - ``training_modules.model``
        - ``model_device``
      * - ``training_modules.loss``
        - ``model_device``
      * - ``training_modules.evaluator.model``
        - ``model_device``
      * - ``controller.optimizer``
        - Partially to ``model_device``
      * - ``database``
        - CPU
      * - Not mentioned
        - CPU

   Again, if you want to load your database to GPU, a manual transfer is
   necessary.

Warning: please do not use something like ``map_location=torch.device(0)``, as
this will map all tensors to GPU 0 and breaks the RNG which only supports a CPU
tensor. Doing so, you will see errors like ``TypeError: RNG state must be a torch.ByteTensor``.
Obviously, moving everything to CPU with ``map_location="cpu"`` always works.
