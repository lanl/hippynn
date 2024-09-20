Hyperparameter optimization with Ax and Ray
===========================================

Here is an example of how you can perform hyperparameter optimization
sequentially (with Ax) or in parallel (with Ax and Ray).

Prerequisites
-------------

The packages required to perform this task are `Ax`_ and `ray`_.
::

    conda install -c conda-forge "ray < 2.7.0"
    pip install ax-platform!=0.4.1

.. note::
   The scripts have been tested with `ax-platform 0.4.0` and `ray 2.6.3`, and
   many previous versions of the two packages. Unfortunately, several changes
   made in recent versions of `ray` will break this script. You should install
   `ray < 2.7.0`. ``pip install`` is recommended by the Ax developers even if
   a conda environment is used.

   As of now (Sep 2024), `ax-platform 0.4.1` is broken. See the `issue`_ here.
   Please avoid this version in your setup.

.. note::
   If you can update this example and scripts to accommodate the changes in the
   latest Ray package, feel free to submit a pull request.

Typical workflow
----------------

Ax is a package that can perform Bayesian optimization. With the given parameter
range, a set of initial trials is generated. Then based on the metrics returned
from these trials, new test parameters are generated. By default, this Ax
workflow can only be performed sequentially. We can combine Ray and Ax to
utilize multiple GPUs on the same node. Ray interfaces with Ax to pull trial
parameters and then automatically distribute the trials to available resources.
With this, we can perform an asynchronous parallelized hyperparameter
optimization.


Create an Ax experiment
^^^^^^^^^^^^^^^^^^^^^^^

You can create a basic Ax experiment this way

.. code-block:: python

    from ax.service.ax_client import AxClient
    ax_client = AxClient()
    ax_client.create_experiment(
        name="hyper_opt",
        parameters=[
            {
                "name": "parameter_a",
                "type": "fixed",
                "value_type": "float",
                "value": 0.6,
            },
            {
                "name": "parameter_b",
                "type": "range",
                "value_type": "int",
                "bounds": [20, 40],
            },
            {
                "name": "parameter_c",
                "type": "choice",
                "bounds": [30, 40, 50, 60, 70],
            },
            {
                "name": "parameter_d",
                "type": "range",
                "value_type": "float",
                "bounds": [0.001, 1],
                "log_scale": True,
            },
        ],
        objectives={
            "Metric": ObjectiveProperties(minimize=True),
        },
        parameter_constraints=[
            "parameter_b <= parameter_c",
        ],
    )

Here we create an Ax experiment called "hyper_opt", with 4 parameters,
`parameter_a`, `parameter_b`, `parameter_c`, and `parameter_d`. Our goal is to
minimize a metric called "Metric".

A few crucial things to note:

* You can give a range, choice, or fixed value to each parameter. You might want
  to specify the data type as well. A fixed parameter makes sense here because
  you can do the optimization with only a subset of parameters without the need
  to modify your training function.
* Constraints can be applied to the search space like the example shows, but
  there is no easy way to achieve a constraint that contains mathematical
  expressions (for example, `parameter_a < 2 * parameter_b`).
* For each experiment, Ax will generate a dictionary as the input of the
  training function. The dictionary will look like::

    {
        "parameter_a": 0.6, 
        "parameter_b": 30,
        "parameter_c": 40,
        "parameter_d": 0.2
    }

  As such, the training function must be able to take a dictionary as the input
  (as a single dictionary or keyword arguments) and use these values to set up
  the training. 
* The `objectives` keyword argument takes a dictionary of variables. The keys of
  the dictionary **MUST** exist in the dictionary returned from the training
  function. In this example, the training function must return a dictionary
  like::

    return {
        ...
        "Metric": metric,
        ...
    }

  The above two points will become more clear when we go through the training
  function.

Training function
^^^^^^^^^^^^^^^^^

You only need a minimal change to your existing training script to use it with
Ax. In most cases, you just have to wrap the whole script into a function

.. code-block:: python

    def training(trial_index, parameter_a, parameter_b, parameter_c, parameter_d):
        # set up the network with the parameters
        ...
        network_params = {
            ...
            "parameter_a": parameter_a,
            ...
        }
        network = networks.Hipnn(
            "hipnn_model", (species, positions), module_kwargs=network_params
        )
        # train the network 
        # `metric_tracker` contains the losses from HIPPYNN
        with hippynn.tools.active_directory(str(trial_index)): 
            metric_tracker = train_model(
            training_modules,
            database,
            controller,
            metric_tracker,
            callbacks=None,
            batch_callbacks=None,
            )
        # return the desired metric to Ax, for example, validation loss
        return {
            "Metric": metric_tracker.best_metric_values["valid"]["Loss"]
        }

Note how we can utilize the parameters passed in and return **Metric** at the
end. Apparently, we have the freedom to choose different metrics to return here.
We can even use mathematical expressions to combine some metrics together.

.. note::
   Ax does NOT create a directory for a trial. If your training function does
   not take care of the working directory, all results will be saved into the
   same folder, i.e., `cwd`. To avoid this, the training function needs to create
   a unique path for each trial. In this example, we use the `trial_index` to
   achieve this purpose. With Ray, this step is NOT necessary.

.. _run-sequential-experiments:

Run sequential experiments
^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we can run the experiments

.. code-block:: python

    for k in range(30):
        parameter, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=training(trial_index, **parameter))
        # Save the experiment as a JSON file
        ax_client.save_to_json_file(filepath="hyperopt.json")
    data_frame = ax_client.get_trials_data_frame().sort_values("Metric")
    data_frame.to_csv("hyperopt.csv", header=True)

For example, we will run 30 trials here and the results will be saved into a
json file and a CSV file. The JSON file will contain all the details of the
trials, which can be used to restart the experiment or add additional trials to
the experiment. As it contains too many details to be human-friendly, we save a
more human-friendly CSV that only contains the trial indices, parameters, and
metrics.

Asynchronous parallelized optimization with Ray
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use Ray to distribute the trials across GPUs parallelly, a small update is
needed for the training function

.. code-block:: python

    from ray.air import session


    def training(parameter_a, parameter_b, parameter_c, parameter_d):
        # setup and train are the same
        # `with hippynn.tools.active_directory() line` is not needed
        ....
        # instead of return, we use `session.report` to communicate with `ray`
        session.report(
        {
            "Metric": metric_tracker.best_metric_values["valid"]["Loss"]
        }
    ) 

Instead of a simple `return`, we need the `report` method from `ray.air.session`
to report the final metric to `ray`.

Also, to run the trials, instead of a loop in :ref:`run-sequential-experiments`,
we have to use the interfaces between the two packages from `ray`

.. code-block:: python

    from ray.tune.experiment.trial import Trial
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.ax import AxSearch

    # to make sure ray loads local packages correctly
    ray.init(runtime_env={"working_dir": "."})

    algo = AxSearch(ax_client=ax_client)
    # 4 GPUs available
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    tuner = tune.Tuner(
        # assign 1 GPU for one trial
        tune.with_resources(training, resources={"gpu": 1}),
        # run 10 trials
        tune_config=tune.TuneConfig(search_alg=algo, num_samples=10),
        # configuration of ray
        run_config=air.RunConfig(
            # all results will be saved in a subfolder inside the "test" folder
            of the current working directory
            local_dir="./test",
            verbose=0,
            log_to_file=True,
        ),
    )
    # run the trials
    tuner.fit()
    # save the results as the end
    # to save the file after each trial, a callback is needed
    # see advanced details
    ax_client.save_to_json_file(filepath="hyperopt.json")
    data_frame = ax_client.get_trials_data_frame().sort_values("Metric")
    data_frame.to_csv("hyperopt.csv", header=True)

This is all you need. The results will be saved in the path of
`./test/{trial_function_name}_{timestamp}`. Each trial will be saved within a
subfolder named
`{trial_function_name}_{random_id}_{index}_{truncated_parameters}`.

Advanced details
^^^^^^^^^^^^^^^^

Relative import
"""""""""""""""

If you save the training function into a separate file and import it into the
Ray script, one line has to be added before the trials start,

.. code-block:: python

   ray.init(runtime_env={"working_dir": "."})

assuming the current directory (".") contains the training and Ray script.
Without this line, Ray will NOT be able to find the training script and import
the training function.

Callbacks for Ray
"""""""""""""""""

When running `ray.tune`, a set of callback functions can be called during the
process. Ray has a `documentation`_ on the callback functions. You can build
your own for your convenience. However, here is a callback function to save
the JSON and CSV files at the end of each trial and handle failed trials, which
should cover the most basic functionalities.

.. code-block:: python

    from ray.tune.logger import JsonLoggerCallback, LoggerCallback
    
    class AxLogger(LoggerCallback):
        def __init__(self, ax_client: AxClient, JSON_name: str, csv_name: str):
            """
            A logger callback to save the progress to a JSON file after every trial ends.
            Similar to running `ax_client.save_to_json_file` every iteration in sequential
            searches.
    
            Args:
                ax_client (AxClient): ax client to save
                json_name (str): name for the JSON file. Append a path if you want to save the \
                    JSON file to somewhere other than cwd.
                csv_name (str): name for the CSV file. Append a path if you want to save the \
                    CSV file to somewhere other than cwd.
            """
            self.ax_client = ax_client
            self.json = json_name
            self.csv = csv_name
    
        def log_trial_end(
            self, trial: Trial, id: int, metric: float, runtime: int, failed: bool = False
        ):
            self.ax_client.save_to_json_file(filepath=self.json)
            shutil.copy(self.json, f"{trial.local_dir}/{self.json}")
            try:
                data_frame = self.ax_client.get_trials_data_frame().sort_values("Metric")
                data_frame.to_csv(self.csv, header=True)
            except KeyError:
                pass
            shutil.copy(self.csv, f"{trial.local_dir}/{self.csv}")
            if failed:
                status = "failed"
            else:
                status = "finished"
            print(
                f"AX trial {id} {status}. Final loss: {metric}. Time taken"
                f" {runtime} seconds. Location directory: {trial.logdir}."
            )
    
        def on_trial_error(self, iteration: int, trials: list[Trial], trial: Trial, **info):
            id = int(trial.experiment_tag.split("_")[0]) - 1
            ax_trial = self.ax_client.get_trial(id)
            ax_trial.mark_abandoned(reason="Error encountered")
            self.log_trial_end(
                trial, id + 1, "not available", self.calculate_runtime(ax_trial), True
            )
    
        def on_trial_complete(
            self, iteration: int, trials: list["Trial"], trial: Trial, **info
        ):
            # trial.trial_id is the random id generated by ray, not ax
            # the default experiment_tag starts with ax' trial index
            # but this workaround is totally fragile, as users can
            # customize the tag or folder name
            id = int(trial.experiment_tag.split("_")[0]) - 1
            ax_trial = self.ax_client.get_trial(id)
            failed = False
            try:
                loss = ax_trial.objective_mean
            except ValueError:
                failed = True
                loss = "not available"
            else:
                if np.isnan(loss) or np.isinf(loss):
                    failed = True
                    loss = "not available"
            if failed:
                ax_trial.mark_failed()
            self.log_trial_end(
                trial, id + 1, loss, self.calculate_runtime(ax_trial), failed
            )
    
        @classmethod
        def calculate_runtime(cls, trial: AXTrial):
            delta = trial.time_completed - trial.time_run_started
            return int(delta.total_seconds())

To use callback functions, simple add a line in ``ray.RunConfig``::

    ax_logger = AxLogger(ax_client, "hyperopt_ray.json", "hyperopt.csv")
    run_config=air.RunConfig(
        local_dir="./test",
        verbose=0,
        callbacks=[ax_logger, JsonLoggerCallback()],
        log_to_file=True,
    )


Restart/extend an experiment
""""""""""""""""""""""""""""

.. note::
   Due to the complexity of handling the individual trial path with Ray, it is
   not possible to restart unfinished trials at this moment.

Restarting an experiment or adding additional trials to an experiment shares the
same workflow. The key is the JSON file saved from the experiment. To reload the 
experiment state:

.. code-block:: python
    
    ax_client = AxClient.load_from_json_file(filepath="hyperopt_ray.json")

Then we can pull new parameters from this experiment, and these parameters will
be generated based on all finished trials. If more trials need to be added to
this experiment, simply increase `num_samples` in `ray.tune.TuneConfig`:

.. code-block:: python

    # this will end the experiment when 20 trials are finished
    tune_config=tune.TuneConfig(search_alg=algo, num_samples=20)

Sometimes, you may want to make changes to the experiment itself when reloading
the experiment, for example, the search space. This can easily achieved by

.. code-block:: python

    ax_client.set_search_space(
        [
            {
                "name": "parameter_b",
                "type": "fixed",
                "value_type": "int",
                "value": 25,
            },
            {
                "name": "parameter_c",
                "type": "choice",
                "values": [30, 40, 50],
            },
        ]
    )

after the `ax_client` object is reloaded.

.. note:: 
   To use the `ax_client.set_search_space` method, the original experiment must
   be created with `immutable_search_space_and_opt_config=False`, i.e.,
       
   .. code-block:: python

       ax_client.create_experiment(
           ...
           immutable_search_space_and_opt_config=False,
           ...
       )
   
   If the original experiment is not created with this option, there is not much
   we can do.

The example scripts with a modified QM7 training script are provided in
`examples`_. This tutorial is contributed by `Xinyang Li`_ and the examples
scripts are developed by `Sakib Matin`_ and `Xinyang Li`_.

.. _ray: https://docs.ray.io/en/latest/
.. _Ax: https://github.com/facebook/Ax
.. _issue: https://github.com/facebook/Ax/issues/2711
.. _documentation: https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html
.. _examples: https://github.com/lanl/hippynn/tree/development/examples/hyperparameter_optimization
.. _Xinyang Li: https://github.com/tautomer
.. _Sakib Matin: https://github.com/sakibmatin
