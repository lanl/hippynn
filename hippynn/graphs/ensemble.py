import collections
from . import GraphModule, replace_node, get_subgraph
from .nodes.misc import EnsembleTarget


def make_ensemble(graphs, targets="auto", inputs="auto", quiet=False):
    """

    :param graphs: list of graph modules, graph modules, or directories specifying models.
     TODO: List of nodes
     TODO: List of directories
     TODO: glob.glob object to define the directories
    :param targets: list of db_name strings or the string 'auto', which will attempt to infer.
    :param inputs: list of db_name strings of the string 'auto', which will attempt to infer.
    :param quiet: whether to print information about the constructed ensemble.
    :return: (ensemble_outputs), (intput_info, output_info),
    """

    graphs = get_models(graphs)

    if inputs == "auto":
        inputs = identify_inputs(graphs)
        if not quiet:
            print("Identified input quantities:", inputs)

    if targets == "auto":
        targets = identify_targets(graphs)
        if not quiet:
            print("Identified output quantities:", inputs)

    input_classes = collate_inputs(graphs, inputs)
    target_classes = collate_targets(graphs, targets)

    ensemble_info = make_ensemble_info(input_classes, target_classes, quiet=quiet)

    ensemble_inputs = replace_inputs(input_classes)
    ensemble_outputs = construct_outputs(target_classes)

    ensemble_graph = make_ensemble_graph(ensemble_inputs, ensemble_outputs)

    return ensemble_graph, ensemble_info


def collate_inputs(models, inputs):
    """

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


def collate_targets(models, targets):
    target_classes = collections.defaultdict(list)

    for m in models:
        for n in m.nodes_to_computer:
            if not hasattr(n, "db_name"):
                continue
            if n.db_name is None:
                continue
            if n.db_name in targets:
                target_classes[n.db_name].append(n)

    target_classes = dict(target_classes.items())

    return target_classes


def get_models(models):
    """

    :param models:
    :return:
    """
    # Todo: replace this with something more advanced.
    # 1) expand glob to dirs and check valid checkpoints
    # 2) load models from dirs
    # convert any nodes to GraphModules.

    return models


def identify_targets(models) -> set[str]:

    targets: set[str] = set()

    for model in models:
        for node in model.nodes_to_compute:
            if node.db_name is not None:
                targets.add(node.db_name)

    return targets


def identify_inputs(models: list[GraphModule]) -> set[str]:

    inputs: set[str] = set()

    for model in models:
        for node in model.input_nodes:
            inputs.add(node.db_name)

    return inputs


def replace_inputs(input_classes):

    ensemble_inputs = []

    for db_name, node_list in input_classes:
        first_node = node_list[0]
        ensemble_inputs.append(first_node)
        rest_nodes = node_list[1:]
        for node in rest_nodes:
            replace_node(node, first_node)

    return ensemble_inputs


def make_ensemble_info(input_classes, output_classes, quiet=False):

    input_info = {k: len(v) for k, v in input_classes()}
    output_info = {k: len(v) for k, v in output_classes.items()}

    if not quiet:
        print("Inputs needed and model counts:")
        for k, v in input_info.items():
            print(f"\t{k}:{v}")
        print("Outputs generated and model counts:")
        for k, v in input_info.items():
            print(f"\t{k}:{v}")

    ensemble_info = input_info, output_info

    return ensemble_info


def construct_outputs(output_classes):

    ensemble_outputs = {}

    for db_name, node_list in output_classes.items():
        parents = node_list
        ensemble_node = EnsembleTarget(name=f"ensemble_{db_name}", parents=parents)
        ensemble_outputs[db_name] = ensemble_node

    return ensemble_outputs


def make_ensemble_graph(ensemble_inputs, ensemble_outputs):

    ensemble_output_list = [c for out in ensemble_outputs for c in out.children]
    ensemble_graph = GraphModule(ensemble_inputs, ensemble_output_list)

    return ensemble_graph
