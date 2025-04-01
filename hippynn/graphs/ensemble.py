import collections
import glob

from ..tools import device_fallback, active_directory
from . import GraphModule, replace_node, get_subgraph

from .indextypes import get_reduced_index_state, index_type_coercion
from .indextypes.reduce_funcs import db_state_of
from .indextypes.registry import assign_index_aliases

from .nodes.base import _BaseNode, InputNode
from .nodes.misc import EnsembleTarget

from .gops import merge_children_recursive

from typing import List, Dict, Union, Tuple


def make_ensemble(
    models,
    *,
    targets: List[str] = "auto",
    inputs: List[str] = "auto",
    prefix: str = "ensemble_",
    quiet=False,
    weights_only=False,
) -> Tuple[GraphModule, Tuple[Dict[str, int], Dict[str, int]]]:

    """

    Make an ensemble out of a set of models. The ensemble graph can then be used with a predictor,
    ase graph, or etc.

    The selected nodes to ensemble are classed by the db_name associated with the nodes.

    When using "auto" mode for inputs and outputs:

    - The input to the ensemble will be the combined inputs for all models in the ensemble.
    - The output of the ensemble will be the combined outputs for all models.

    Otherwise, the set of ensemble inputs is explicitly specified, and errors
    may occur if the requested set of inputs and outputs is not available.

    The result ensemble graph has several outputs, each of which has .mean, .std, and .all
    attributes which reflect the statistics of the models in the ensemble.

    Note that it is not required that all models have the same sets of inputs and outputs.

    If a desired node is automatically ensembled, it probably does not have a db_name.
    A remedy for this is to load the graphs with hippynn.graphs.ensemble.get_graphs, then
    find the requested nodes in the graphs and assign them the db_name. Then pass these
    graphs to make_ensemble.

    For more information on the `models` parameter, see the :func:`~hippynn.graphs.ensemble.get_graphs` function.

    .. Warning::
        This function uses ``torch.load`` with ``weights_only=False`` by default,
        which can run arbitrary code from the loaded file. Only use it with files
        you trust. You can set ``weights_only=True`` for better security, but
        it may cause loading of the structure file to fail. 

    :param models: list containing str, node, or graphmodule, or str to glob for model directories.
    :param targets: list of db_name strings or the string 'auto', which will attempt to infer.
    :param inputs: list of db_name strings of the string 'auto', which will attempt to infer.
    :param prefix: specifies the prefix for the db_name of created ensemble nodes.
    :param quiet: whether to print information about the constructed ensemble.
    :param weights_only: passed to ``torch.load``
    :return: ensemble GraphModule, (intput_info, output_info)
    """

    # Phase 0: Make sure we are dealing with GraphModules
    graphs: List[GraphModule] = get_graphs(models, weights_only=weights_only)

    # Phase 1: Figure out what the ensemble will look like.
    if inputs == "auto":
        inputs: set[str] = identify_inputs(graphs)
        if not quiet:
            print("Identified input quantities:", inputs)

    if targets == "auto":
        targets: set[str] = identify_targets(graphs)
        if not quiet:
            print("Identified output quantities:", targets)

    input_classes: Dict[str, List[_BaseNode]] = collate_inputs(graphs, inputs)
    target_classes: Dict[str, List[_BaseNode]] = collate_targets(graphs, targets)

    ensemble_info = make_ensemble_info(input_classes, target_classes, quiet=quiet)

    # Phase 2 build ensemble graph and GraphModule.
    ensemble_outputs: Dict[str, EnsembleTarget] = construct_outputs(target_classes, prefix=prefix)
    ensemble_inputs: List[_BaseNode] = replace_inputs(input_classes)
    merged_inputs: List[_BaseNode] = merge_children_recursive(ensemble_inputs)

    if not quiet:
        print("Merged the following nodes from the ensemble members:")
        for node in merged_inputs:
            print("\t", node)

    ensemble_graph = make_ensemble_graph(ensemble_inputs, ensemble_outputs)

    return ensemble_graph, ensemble_info


# TODO: Potentially move this function, or part of it, into experiment.serialization?
# TODO ; It seems possible that someone might want to load several models without ensembling them.
def get_graphs(models: Union[List[Union[str, GraphModule, _BaseNode]], str], weights_only: bool = False) -> List[GraphModule]:
    """
    Take a simple spec for modeled variables (glob for model directories, list of graphs, list of output nodes)
    and convert this into a list of graph modules.


    Models can be a list with entries that are one of the following:

    - str: directory to use with :func:`~hippynn.experiment.serialization.load_model_from_cwd`
    - GraphModule: already built model
    - node: an output target, which will be converted to a GraphModule with automatically defined inputs.

    or a string, which is used with glob to specify the list of strings.

    .. Warning::
        This function uses ``torch.load`` with ``weights_only=False`` by default,
        which can run arbitrary code from the loaded file. Only use it with files
        you trust. You can set ``weights_only=True`` for better security, but
        it may cause loading of the structure file to fail. 

    :param models:
    :param weights_only: passed to ``torch.load``
    :return:
    """

    graphs = []
    if isinstance(models, str):
        models = glob.glob(models)

    device = None
    for model in models:
        if isinstance(model, str):
            from ..experiment.serialization import load_model_from_cwd

            # Get graph from disk
            if device is None:
                device = device_fallback()
            with active_directory(model, create=False):
                try:
                    model = load_model_from_cwd(map_location=device, weights_only=weights_only)
                except FileNotFoundError:
                    import warnings

                    warnings.warn(f"Model not found in directory: {model}")
                else:
                    graphs.append(model)

        elif isinstance(model, _BaseNode):
            subgraph = get_subgraph([model])
            subgraph_inputs = list({x for x in subgraph if isinstance(x, InputNode)})
            model = GraphModule(subgraph_inputs, [model.main_output])
            graphs.append(model)

        elif isinstance(model, GraphModule):
            graphs.append(model)

    return graphs


def identify_targets(models: List[GraphModule]) -> set[str]:
    """
    Internal function for ensembling.

    Identify targets types to ensemble.

    :param models:
    :return:
    """

    targets: set[str] = set()

    for model in models:
        for node in model.nodes_to_compute:
            if node.db_name is not None:
                targets.add(node.db_name)

    return targets


def identify_inputs(models: list[GraphModule]) -> set[str]:
    """
    Internal function for ensembling.

    Find all required inputs for ensemble.

    :param models:
    :return:
    """

    inputs: set[str] = set()

    for model in models:
        for node in model.input_nodes:
            inputs.add(node.db_name)

    return inputs


def collate_inputs(models: list[GraphModule], inputs: List[str]) -> Dict[str, List[_BaseNode]]:
    """
    Internal function for ensembling.

    Identify input classes for ensemble.

    :param models:
    :param inputs:
    :return:
    """
    input_classes = collections.defaultdict(list)

    for m in models:
        for n in m.input_nodes:
            if n.db_name not in inputs:
                raise ValueError("Input not allowed: '{n.db_name}' (Allowed targets were {inputs}")
            input_classes[n.db_name].append(n)

    input_classes = dict(input_classes.items())
    return input_classes


def collate_targets(models: List[GraphModule], targets: List[str]) -> Dict[str, List[_BaseNode]]:
    """
    Internal function for ensembling.

    Identify targets to ensemble.

    :param models:
    :param targets:
    :return:
    """
    target_classes = collections.defaultdict(list)

    for m in models:
        for n in m.nodes_to_compute:
            if not hasattr(n, "db_name"):
                continue
            if n.db_name is None:
                continue
            if n.db_name in targets:
                target_classes[n.db_name].append(n)

    target_classes = dict(target_classes.items())

    return target_classes


def make_ensemble_info(input_classes: Dict[str, List[_BaseNode]], output_classes: Dict[str, List[_BaseNode]], quiet=False):
    """
    Internal function for ensembling.

    Count up and print the ensemble variables identified.

    :param input_classes:
    :param output_classes:
    :param quiet:
    :return:
    """

    input_info = {k: len(v) for k, v in input_classes.items()}
    output_info = {k: len(v) for k, v in output_classes.items()}

    if not quiet:
        print("Inputs needed and respective model counts:")
        for k, v in input_info.items():
            print(f"\t{k} : {v}")
        print("Outputs generated and respective model counts:")
        for k, v in output_info.items():
            print(f"\t{k} : {v}")

    ensemble_info = input_info, output_info

    return ensemble_info


def construct_outputs(output_classes: Dict[str, List[_BaseNode]], prefix: str) -> Dict[str, EnsembleTarget]:
    """
    Internal function for ensembling.

    Build the EnsembleNodes for classes of outputs. Note that
    this function attempts to produce consistent index state versions
    for both a database-type index state and a reduced index state.

    :param output_classes: Dictionary giving list of nodes to ensemble.
    :param prefix: name to prepend to each ensembled variable.
    :return: dictionary of ensembled node names to ndoes.
    """

    # To facilitate conversion of index states of ensembled nodes, we will build
    # an ensemble target for both the db_form and the reduced form for each node.
    # The ensemble will return the db_form when they differ,
    # but the index cache will still register the reduced form (when it is different)
    # Note!: We want to run these before linking the separate models together,
    # because the automation algorithms of hippynn currently handle cases
    # where there is a unique type for some nodes in the graph, e.g. one pair indexer
    # or one padding indexer.
    # Therefore, in the first loop, we generate all index states before any
    # of the ensemble nodes have been constructed.

    for db_name, parents in sorted(output_classes.items(), key=lambda x: x[0]):
        reduced_index_state = get_reduced_index_state(*parents)
        db_index_state = db_state_of(reduced_index_state)
        db_state_parents = [index_type_coercion(p, db_index_state) for p in parents]
        reduced_parents = [index_type_coercion(p, reduced_index_state) for p in parents]

    # Now that the index states for each node have been generated and cached,
    # we can loop through again and build the ensembled target nodes.

    ensemble_outputs = {}
    for db_name, parents in sorted(output_classes.items(), key=lambda x: x[0]):

        reduced_index_state = get_reduced_index_state(*parents)
        db_index_state = db_state_of(reduced_index_state)

        db_state_parents = [index_type_coercion(p, db_index_state) for p in parents]
        reduced_parents = [index_type_coercion(p, reduced_index_state) for p in parents]

        # Build db_form output
        ensemble_node = EnsembleTarget(name=f"{prefix}{db_name}", parents=db_state_parents)
        ensemble_outputs[db_name] = ensemble_node

        if reduced_index_state != db_index_state:
            name = f"ensemble_{db_name}[{reduced_index_state}]"

            reduced_ensemble_node = EnsembleTarget(name=name, parents=reduced_parents)

            for db_child, reduced_child in zip(ensemble_node.children, reduced_ensemble_node.children):
                assign_index_aliases(db_child, reduced_child)

    return ensemble_outputs


def replace_inputs(input_classes: Dict[str, List[_BaseNode]]) -> List[InputNode]:
    """
    Internal function for ensembling.

    Replace all input nodes with the first representative from the class, thereby
    merging inputs.

    :param input_classes:
    :return: list of new input nodes.
    """

    ensemble_inputs = []

    for db_name, node_list in input_classes.items():
        first_node = node_list[0]
        ensemble_inputs.append(first_node)
        rest_nodes = node_list[1:]
        for node in rest_nodes:
            replace_node(node, first_node)

    return ensemble_inputs


def make_ensemble_graph(ensemble_inputs: List[InputNode], ensemble_outputs: Dict[str, EnsembleTarget]) -> GraphModule:
    """
    Internal function for ensembling.

    Put together the ensemble graph.

    :param ensemble_inputs:
    :param ensemble_outputs:
    :return:
    """

    ensemble_output_list = [c for k, out in ensemble_outputs.items() for c in out.children]
    ensemble_graph = GraphModule(ensemble_inputs, ensemble_output_list)

    return ensemble_graph
