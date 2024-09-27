Plotting
========


:mod:`hippynn.plotting` is only available if matplotlib is installed.

By default, hippynn will plot loss metrics over time when training ends.
On top of this, hippynn can make diagnostic plots during its evaluation phase.
For example, Let's assume you have a ``molecule_energy`` node that you are training to.
A simple plot maker would look like this::

    from hippynn import plotting

    plot_maker = hippynn.plotting.PlotMaker(
        plotting.Hist2D.compare(molecule_energy, saved=True,shown=False),
        plot_every=10
    )

    from hippynn.experiment import assemble_for_training

    training_modules,db_info = assemble_for_training(train_loss, validation_losses, plot_maker=plot_maker)

The plot maker is thus passed to :func:`~hippynn.experiment.assemble_for_training` and attached to the model evaluator.




