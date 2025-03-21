"""
Core GraphModule that implements node computations
"""
import warnings
from collections import OrderedDict

import torch

from .nodes.base.node_functions import _BaseNode
from .nodes.base import InputNode

from . import get_subgraph, compute_evaluation_order

from .. import settings


# possible TODO: Enable "virtual input nodes" where we accept the value for any given node
# Requires copying subgraph, eliminating the parents which belong only to the
# Virtual input nodes, and then creating an input node for each virtual input node
# Idea: allow someone to evaluate parts of the graph when other parts are known
# implementation: use swap_parent and generate input nodes with the right index
# type.


class GraphModule(torch.nn.Module):
    def __init__(self, required_inputs, nodes_to_compute):
        """

        takes a graph, makes a pytorch neural network

        :param required_inputs: inputs
        :param nodes_to_compute: outouts
        """
        super().__init__()

        nodes_to_compute = list(nodes_to_compute)

        if len(nodes_to_compute) == 0:
            raise ValueError("Length of `nodes_to_compute` was zero. A graph module "
                             "must receive a list of outputs with length greater than zero.")

        assert all(isinstance(n, InputNode) for n in required_inputs)

        self.input_nodes = required_inputs
        self.nodes_to_compute = nodes_to_compute

        all_node_list = get_subgraph(nodes_to_compute)

        for node in all_node_list:
            if isinstance(node, InputNode):
                if node not in self.input_nodes:
                    raise ValueError("Nodes to compute requires an unspecified input:", node)

        for x in all_node_list:
            if not isinstance(x, _BaseNode):
                if not callable(x):
                    raise ValueError("Computational objects must be callable.")
                else:
                    warnings.warn("Warning: object not base node:", x)

        del x

        self.forward_output_list, self.forward_inputs_list = compute_evaluation_order(all_node_list)

        # Here we have to do a small dance to allow us to create a dictionary of nodes to modules.
        # Pytorch requires ModuleDicts to have keys that are strings.
        # We want a ModuleDict with keys that are nodes.
        # self.node_mod_map does this as a regular python dictionary,
        # while registration with pytorch is given by numbering.

        self.names_dict = OrderedDict((n, "node" + str(i)) for i, n in enumerate(self.forward_output_list))
        self.moddict = torch.nn.ModuleDict(
            OrderedDict(
                (
                    (node_name, node.torch_module)
                    for node, node_name in self.names_dict.items()
                    if not isinstance(node, InputNode)
                )
            )
        )

    def get_module(self, node):
        return self.moddict[self.names_dict[node]]

    def print_structure(self, suppress=True):
        """Pretty-print the structure of the nodes and links comprising this graph."""
        in_nodes = {n: "I{}".format(i) for i, n in enumerate(self.input_nodes)}
        out_nodes = {n: "O{}".format(i) for i, n in enumerate(self.nodes_to_compute)}
        middle_nodes = {n: "H{}".format(i) for i, n in enumerate(self.forward_output_list)}
        middle_nodes.update(out_nodes)
        node_map = {**in_nodes, **middle_nodes}
        print("Inputs:")
        for k, v in in_nodes.items():
            print("\t", v, ":", k)
        print("Outputs:")
        for k, v in out_nodes.items():
            print("\t", v, ":", k)
        print("Order:")
        all_inputs = set(n for this_list in self.forward_inputs_list for n in this_list)
        for computed, inputs_for_computed in zip(self.forward_output_list, self.forward_inputs_list):
            if computed not in all_inputs and computed not in self.nodes_to_compute:
                # most likely this is just the child of MultiNode which is still unpacked.
                continue

            pre = ",".join({node_map[n] for n in inputs_for_computed})
            mid = "{:3} : {}".format(node_map[computed], computed.name)
            print("{:-<20}-> {}".format(pre, mid))

        node_map.update()
        node_map.update()
        node_map.update({n: "Out{}:".format(i) for i, n in enumerate(self.forward_output_list)})

    def node_from_name(self, name):
        for match in list(self.input_nodes) + list(self.forward_output_list):
            if match.db_name == name or match.name == name:
                node = match
                break
        else:
            raise ValueError("Name '{}' not found in graph.".format(name))
        return node

    def extra_repr(self):
        return "Inputs: {} \n Outputs: {}".format(
            tuple(x.name for x in self.input_nodes), tuple(x.name for x in self.nodes_to_compute)
        )

    def forward(self, *input_values):
        # Add gradient computation if needed (e.g. for force computation)
        computed = {
            node: value.requires_grad_(value.requires_grad or node.requires_grad)
            for node, value in zip(self.input_nodes, input_values)
        }

        for this_node, inputs_for_this_node in zip(self.forward_output_list, self.forward_inputs_list):
            computed[this_node] = self.get_module(this_node)(*(computed[inkey] for inkey in inputs_for_this_node))

        return tuple(computed[x] for x in self.nodes_to_compute)


class _DebugGraphModule(GraphModule):
    def forward(self, *input_values):
        try:
            computed = {
                node: value.requires_grad_(value.requires_grad or node.requires_grad)
                for node, value in zip(self.input_nodes, input_values)
            }
        except Exception as ee:
            for n, v in zip(self.input_nodes, input_values):
                print(
                    "\tNode:",
                    n,
                )
                print("\t\tValue:", v)
            raise Exception("Something broke loading the values.") from ee

        print("INPUTTED:")
        for k, v in computed.items():
            print("\t", k.name, v.device, v.shape, v)
        print("Should be inputted:")
        print(*(i.name for i in self.input_nodes), sep="\n\t")

        for this_node, inputs_for_this_node in zip(self.forward_output_list, self.forward_inputs_list):
            print("Computing:", this_node.name)
            try:
                inputs = tuple(computed[inkey] for inkey in inputs_for_this_node)
                module = self.get_module(this_node)
                print("module:", module)
                print("Inputs for this node:", [x.name for x in inputs_for_this_node])
                print("input devices:", [(x.device if isinstance(x, torch.Tensor) else type(x)) for x in inputs])
                value = module(*inputs)
                # print("inputs:",inputs)
                # print("Value:",value)
                if isinstance(value, torch.Tensor):
                    print("Output shape:", value.shape, value.dtype)
                    if torch.isnan(value).any():
                        raise ValueError("Nan detected! {}".format(value))
                else:
                    print("Output non-tensor.")
                computed[this_node] = value
            except Exception as e:
                print("Problem!")
                print("Inputs for this node:", [x.name for x in inputs_for_this_node])
                for in_node in inputs_for_this_node:
                    in_comp = computed[in_node]
                    if isinstance(in_comp, torch.Tensor):
                        print("\t", in_node.name, in_comp.dtype, in_comp.shape)
                    else:
                        print("\t non-tensor:", in_comp)

                print(e)
                print("Computing state:")
                print({k.name: k for k in computed.keys()})
                # for k,v in computed.items():
                #    print("Name: {} Value: {}".format(k.name,v))
                raise e

        return tuple(computed[x] for x in self.nodes_to_compute)


if settings.DEBUG_GRAPH_EXECUTION:
    GraphModule = _DebugGraphModule
