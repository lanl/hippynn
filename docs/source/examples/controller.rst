Controller
==========


How to define a controller for more customized control of the training process.
We assume that there is a set of ``training_modules`` assembled and a ``database`` object has been constructed.

The following snippet shows how to set up a controller using a custom scheduler or optimizer::

    from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau,PatienceController

    optimizer = torch.optim.Adam(training_modules.model.parameters(),lr=1e-3)

    scheduler =  RaiseBatchSizeOnPlateau(optimizer=optimizer,
                                         max_batch_size=80,
                                         patience=5,
                                         max_epochs=200)

    controller = PatienceController(optimizer=optimizer,
                                    scheduler=scheduler,
                                    batch_size=10,
                                    eval_batch_size=512,
                                    max_epochs=1000,
                                    termination_patience=20,
                                    fraction_train_eval=0.1,
                                    stopping_key=early_stopping_key,
                                    )

    experiment_params = hippynn.experiment.SetupParams(
        controller = controller,
        device='cpu'
    )

The role of the controller is to govern the dynamics of training outside of the concerns
of the optimizer (how to perform parameter updates) and the scheduler (how to modify the
optimizer). The controller reports when to stop training.
