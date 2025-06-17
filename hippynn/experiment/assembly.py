"""
Assembling graphs for a training run
"""
import copy
import warnings
import collections

from hippynn.graphs import GraphModule, get_subgraph, find_unique_relative
from hippynn.graphs.nodes.base import InputNode, LossInputNode, LossPredNode, LossTrueNode

from hippynn.experiment.evaluator import Evaluator

TrainingModules = collections.namedtuple("TrainingModules", ("model", "loss", "evaluator"))
"""
:param model: assembled torch.nn.Module of the model
:param loss: assembled torch.nn.Module for the loss
:param evaluator: assembled evaluator for validation losses
"""


def generate_database_info(inputs, targets, allow_unfound=False):
    """
    Construct db info from input nodes and target nodes.
    :param inputs: list of input nodes
    :param targets: list of target nodes
    :param allow_unfound: don't check if names are valid

    Builds a list of the db names for the nodes.
    If

    :return:
    """
    db_info = {
        "inputs": [i.db_name for i in inputs],
        "targets": [i.db_name for i in targets],
    }

    # Allows `None` to pass through.
    if allow_unfound:
        return db_info

    # If none of the names was `None`, return.
    if not any(name is None for row in db_info.values() for name in row):
        return db_info

    # Else, we need to raise an error.

    missing_inputs = [i for i, idb in zip(inputs, db_info["inputs"]) if idb is None]
    missing_targets = [i for i, idb in zip(targets, db_info["targets"]) if idb is None]

    msg = ""
    if missing_inputs:
        msg += "Missing inputs: {}\n".format(missing_inputs)
    if missing_targets:
        msg += "Missing targets: {}\n".format(missing_targets)

    raise ValueError("Model/Loss configuration requires database quantity to be defined for these factors:\n{}".format(msg))


def build_loss_modules(training_loss, validation_losses, network_outputs, database_inputs):
    network_outputs = [x.pred for x in network_outputs]
    database_inputs = [x.true for x in database_inputs]
    all_inputs = (*network_outputs, *database_inputs)

    train_loss_module = GraphModule(all_inputs, [training_loss])
    valid_loss_module = GraphModule(all_inputs, validation_losses)
    valid_loss_module = copy.deepcopy(valid_loss_module)

    return train_loss_module, valid_loss_module


def determine_out_in_targ(*nodes_required_for_loss):
    """
    :param nodes_required_for_loss: train nodes and validation nodes
    :return: lists of needed network inputs, network outputs, and database targets
    """

    nodes_required_for_loss = get_subgraph(nodes_required_for_loss)

    required_inputs = {x for x in nodes_required_for_loss if isinstance(x, InputNode)}

    assert all(isinstance(x, LossInputNode) for x in required_inputs), (
        "Losses should not contain plain InputNode objects; only LossPredNode and LossTrueNode objects"
        "Bad nodes: {} ".format([x for x in required_inputs if not isinstance(x, LossInputNode)])
    )

    outputs = [x.origin_node for x in required_inputs if isinstance(x, LossPredNode)]
    targets = [x.origin_node for x in required_inputs if isinstance(x, LossTrueNode)]
    inputs = [p for node in outputs for p in node.get_all_parents() if isinstance(p, InputNode)]

    inputs = list(set(inputs))
    outputs = list(set(outputs))
    targets = list(set(targets))

    return inputs, outputs, targets


def assemble_for_training(train_loss, validation_losses, validation_names=None, plot_maker=None):
    """
    :param train_loss: LossNode
    :param validation_losses: dict of (name:loss_node) or list of LossNodes.
                              -if a list of loss nodes, the name of the node will be used for printing the loss,
                              -this can be overwritten with a list of validation_names
    :param validation_names (optional):  list of names for loss nodes, only if validation_losses is a list.
    :param plot_maker:        optional PlotMaker for model evaluation
    :return: training_modules, db_info
        -db_info: dict of inputs (input to model) and targets (input to loss) in terms of the `db_name`.

    ``assemble_for_training`` computes:

    #. what inputs are needed to the model
    #. what outputs of the model are needed for the loss
    #. what targets are needed from the database for the loss

    It then uses this info to create ``GraphModule`` s for the model and loss,
    and an ``Evaluator`` based on validation loss (& names), early stopping, plot maker.

    .. Note::
        Model and training loss are always evaluated on the active device.
        But the validation losses reside by default on the CPU. This helps compute statistics over large datasets.
        To accomplish this, the modules associated with the loss are copied in the validation loss.
        Thus, after assembling the modules for training, changes to the loss nodes will not affect the model evaluator.
        In all likelihood you aren't planning to do something too fancy like change the loss nodes during training.
        But if you do plan to do something like that with callbacks, know that you would probably need to construct a new
        evaluator.

    """
    if validation_names is None:
        if isinstance(validation_losses, dict):
            validation_names = list(validation_losses.keys())
            validation_losses = list(validation_losses.values())
        else:
            validation_names = [n.name for n in validation_losses]
    else:
        assert not isinstance(
            validation_losses, dict
        ), "Validation loss names cannot be supplied if validation_losses is a dictionary"

    if not all(isinstance(key, str) for key in validation_names):
        raise ValueError("Validation names must be strings.")

    loss_required_nodes = (train_loss, *validation_losses)

    if plot_maker is not None:
        loss_required_nodes = *loss_required_nodes, *plot_maker.required_nodes

    inputs, outputs, targets = determine_out_in_targ(*loss_required_nodes)

    print("Determined Inputs:", [x.name for x in inputs])
    print("Determined Outputs:", [x.name for x in outputs])
    print("Determined Targets:", [x.name for x in targets])

    model = GraphModule(inputs, outputs)

    loss_assembled, validation_lossfns = build_loss_modules(train_loss, validation_losses, outputs, targets)

    if plot_maker is not None:
        plot_maker.assemble_module(outputs, targets)

    db_info = generate_database_info(inputs, targets)

    evaluator = Evaluator(model, validation_lossfns, validation_names, plot_maker=plot_maker, db_info=db_info)

    return TrainingModules(model, loss_assembled, evaluator), db_info


_PAIRCACHE_DB_NAME = "AutoPrecomputedPairs"
from ..graphs.nodes import pairs, indexers, base
from ..graphs import Predictor, gops, inputs

_PAIRCACHE_COMPATIBLE_COMPUTERS = {pairs.NumpyDynamicPairs, pairs.PeriodicPairIndexer, pairs.DynamicPeriodicPairs}


def precompute_pairs(model, database, batch_size=10, device=None, make_dense=False, n_images=1):
    """

    :param model: Assembled GraphModule involving a PairIndexer
    :param database: database that precomputation should be supplied with
    :param batch_size: batch size to do pre-computation with
    :param device: where to do the precomputation.
    :param make_dense: return a dense array of pairs. Warning, this can be memory-expensive. However, it is necessary if you
        are going to use num_workers>0 in your dataloaders. If False, the cache is stored as a sparse array.
    :param n_images: number of images for cache storage; increase this if it fails.
        However, large values can incur a  very large memory cost if make_dense is True.

    :return: None-- changes the model graph.

    .. note ::
       After running pre-compute pairs, your model will expect to load pairs directly from the database,
       and your database will contain cached pair entries.

    Note that the returned model needs to be re-assembled with the new graph for the cache to take effect.
    Example usage:
    >>> precompute_pairs(training_modules.model,database,device='cuda')
    >>> training_modules, db_info = assemble_for_training(train_loss, validation_losses)
    >>> database.inputs = db_info['inputs']
    """

    # nodes_to_compute,all_copies = gops.copy_subgraph(model.nodes_to_compute,assume_inputed=[])
    nodes_to_compute = model.nodes_to_compute
    pair_indexer = find_unique_relative(
        nodes_to_compute, lambda node: isinstance(node, tuple(_PAIRCACHE_COMPATIBLE_COMPUTERS))
    )
    dist_hard_max = pair_indexer.dist_hard_max
    cacher = pairs.PairCacher("PairCacher", pair_indexer, module_kwargs=dict(n_images=n_images))

    input_nodes = set([x for x in cacher.get_all_parents() if isinstance(x, base.InputNode)])
    pred = Predictor(input_nodes, [cacher], model_device=device, name="Pair Precomputer")

    try:
        for k, arrdict in database.splits.items():
            input_arrs = {node: arrdict[node.db_name] for node in input_nodes}

            outputs = pred(node_values=input_arrs, batch_size=batch_size)
            cache = outputs[cacher]
            if make_dense:
                cache = cache.to_dense()
            else:
                cache = cache.coalesce()
            database.splits[k][_PAIRCACHE_DB_NAME] = cache
    except RuntimeError as re:
        msg = re.args[0]
        if "size is inconsistent with indices" in msg:
            raise ValueError("Caching pairs required a larger number of images `n_images`")
        else:
            raise re

    cacheinput = inputs.PairIndices(db_name=_PAIRCACHE_DB_NAME)
    pos = pair_indexer.coordinates
    cell = pair_indexer.cell
    atomidx = gops.find_unique_relative(pair_indexer, indexers.AtomIndexer)
    uncacher = pairs.PairUncacher("AutoPairUncache",
                                  (cacheinput, pos, cell, atomidx),
                                  dist_hard_max=dist_hard_max)
    gops.replace_node(pair_indexer, uncacher, disconnect_old=True)

    return


# def reassemble(training_modules):
#     raise ValueError("Broken because of copying of loss graph for evaluator!")
#     model, loss_graph, evaluator=training_modules
#     train_loss = loss_graph.nodes_to_compute[0]
#     validation_losses = evaluator.loss.nodes_to_compute
#     validation_names = evaluator.loss_names
#     plot_maker = evaluator.plot_maker
#
#     return assemble_for_training(train_loss=train_loss,
#                                  validation_losses=validation_losses,
#                                  validation_names=validation_names,
#                                  plot_maker=plot_maker)
