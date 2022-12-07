Units in hippynn
================

For the most part, hippynn is designed to operate *transparently* with
respect to units, meaning that values that have units should be specified
in the units of the dataset that is being examined.

The primary example of this is the sensitivity functions. Network hyperparameters
often involve a hard distance cutoff for local interactions. It's natural to ask:
what units does that cutoff parameter need to be in? What units should you put
your data in?

The philosophy of unit transparency means that the input units in a dataset
should be the same as the units in the hyperparameters, but which units you
use are not in themselves important.

However, there are still effects in training that can be relative to the
unit system of choice. This is because the value of network outputs
has an intrinsic scale tied to the choice of weight initialization
and activation function. For example, if you want to predict a feature whose typical
values are very small, say on order :math:`10^{-4}`, this will likely lead to a need for
weights that are numerically much smaller than the initial weights in the network.
This can be alleviated in two ways. One is to just modify the database to work with a
different unit scale, such that the prediction target is order one. Another is
to just make a new node which incorporates the effective scale into the prediction::

    from hippynn.graphs import inputs, networks, targets
    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")
    network = networks.Hipnn("hipnn_model", (species, positions), module_kwargs = network_params)
    hcharge = targets.HChargeNode("hcharge",network)
    scaled_charges = 1e-4 * hcharge.atom_charges
    scaled_charges.db_name = "charge"

This model will thus not need to have very small network weights to make very small
predictions.

The other way that units can play a role is in the construction of the loss function.
For example, with very small target values, it may be possible for
the Mean Squared Error to be far smaller than a regularization term. As such
it is called for to weight the various terms going into the loss function so that they
contribute to the loss in the desired manner; for example, one often wants regularization
terms to be a small component of the total loss. When training to multiple targets,
one usually wants them to contribute, in the final trained model, roughly equally
to the total loss.



Similar comments, but reversed, play a role if your targets have very large values.
