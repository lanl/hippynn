Profiling 
==============================

The :func:`~hippynn.experiment.setup_and_profile` function is a drop-in replacement 
for :func:`~hippynn.experiment.setup_and_train` that profiles your training loop 
using PyTorch's built-in profiler.


Simply swap ``setup_and_train`` with ``setup_and_profile``::

    # Before:
    from hippynn.experiment import setup_and_train

    setup_and_train(
        training_modules=training_modules,
        database=database,
        setup_params=experiment_params,
    )

    # After:
    from hippynn.experiment import setup_and_profile

    setup_and_profile(
        training_modules=training_modules,
        database=database,
        setup_params=experiment_params,
    )


By default, the function runs only 3 epochs with 5 batches each, temporarily overriding 
your ``max_epochs`` setting. This provides enough data to capture representative performance 
patterns without the overhead of a full training run.

The :func:`~hippynn.experiment.setup_and_profile` function accepts optional parameters 
not available in ``setup_and_train`` to control the profiling behavior::

    setup_and_profile(
        training_modules=training_modules,
        database=database,
        setup_params=experiment_params,
        profile_epochs=5,              # Optional: defaults to 3
        batches_per_epoch=10,          # Optional: defaults to 5
        trace_file="my_trace.json",    # Optional: defaults to "profile_trace.json"
    )



After profiling completes, a summary table is printed to the console showing the 
most time-consuming operations. Additionally, a JSON trace file is saved to disk.

To visualize the detailed timeline, open ``chrome://tracing`` in Google Chrome (must be chrome, not firefox or safari) 
and load the JSON file. 
