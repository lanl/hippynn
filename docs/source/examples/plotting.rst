Plotting
========


How to make a plotmaker.

Let's assume you have a `molecule_energy` node that you are training to.
A simple plot maker would look like this::


    from hippynn import plotting

    plot_maker = hippynn.plotting.PlotMaker(
        plotting.Hist2D.compare(molecule_energy, saved=True,shown=False),
        plot_every=10
    )

    from hippynn.experiment import assemble_for_training

    training_modules,db_info = assemble_for_training(train_loss, validation_losses, plot_maker=plot_maker)

The plot maker is thus passed to `assemble_for_training` and attached to the model evaluator.