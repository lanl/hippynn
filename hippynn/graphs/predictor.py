"""
Predictor object which builds a graph for predictions with
a simple user interface.
"""
import torch

from ..tools import arrdict_len, progress_bar

from .graph import GraphModule
from .gops import search_by_name
from .indextypes import IdxType, db_form

# TODO: batched prediction for scalars
# It doesn't work for scalars! Right now they are just filtered out.


class Predictor:
    """
    The predictor is a dressed-up GraphModule which gives access
    to the outputs of individual nodes.

    In many cases you may simply want to use the ``from_graph`` method to generate a predictor.

    The predictor will take the model graph, convert the output nodes into
    a padded index state, and build a new graph for these operations.

    """

    def __init__(self, inputs, outputs, return_device=torch.device("cpu"), model_device=None, requires_grad=False,
                 name=None):
        """

        :param inputs: nodes to use as inputs
        :param outputs: nodes to use as outputs
        :param return_device: device to place results on; does nothing if None
        :param model_device: where the model and input should be placed, does nothing if None
        :param requires_grad: Default false -- detach predicted tensors. If true, do not detach predicted
         tensors -- this may lead to memory leaks if not done carefully.
        :param name: Optional name for formatting progress bars.

        """

        outputs = [search_by_name(inputs, o) if isinstance(o, str) else o for o in outputs]
        outputs = list(set(outputs))  # Remove any redundancies -- they will screw up the output name map.

        outputs = [o for o in outputs if o._index_state is not IdxType.Scalar]

        self.out_names = [o.name for o in outputs]
        self.out_dbnames = [o.db_name for o in outputs]

        outputs = [db_form(o) for o in outputs]
        self.graph = GraphModule(inputs, outputs)

        # User can supply kwargs using the node's name or db_name, the latter takes priority
        self.argmap = {
            **{node.name: node for node in self.graph.input_nodes},
            **{node.db_name: node for node in self.graph.input_nodes},
        }

        self.return_device = return_device
        self.model_device = model_device
        self.requires_grad = requires_grad
        self.name = name or "Predictor"

    @classmethod
    def from_graph(cls, graph, additional_outputs=None, **kwargs):
        """
        Construct a new predictor from an existing GraphModule.

        :param graph:  graph to create predictor for. The predictor makes a shallow copy of this graph. e.g. it may move
           parameters from that graph to the model_device.
        :param additional_outputs: List of additional nodes to include in outputs
        :param kwargs: passed to ``__init__``

        :return: predictor instance
        """
        inputs = graph.input_nodes
        outputs = graph.nodes_to_compute
        if additional_outputs is not None:
            outputs = outputs + list(additional_outputs)

        return cls(inputs, outputs, **kwargs)

    def to(self, *args, **kwargs):
        return self.graph.to(*args, **kwargs)

    @property
    def inputs(self):
        return self.graph.input_nodes

    @property
    def outputs(self):
        return self.graph.nodes_to_compute

    @property
    def model_device(self):
        return self._model_device

    @model_device.setter
    def model_device(self, device):
        self.graph.to(device)
        self._model_device = device

    def add_output(self, node):
        if isinstance(node, str):
            node = self.graph.node_from_name(node)

        source = node
        node = db_form(node)
        if node not in self.graph.forward_output_list:
            raise ValueError(
                "Node {} is not computed in graph. We suggest you make a ".format(node)
                + "new predictor with this node in the additional_outputs list."
            )

        self.outputs.append(node)
        self.out_names.append(source.name)
        self.out_dbnames.append(source.bname)

    def wrap_outputs(self, out_dict):
        for (node, tensor), dbname, name in zip(out_dict.copy().items(), self.out_dbnames, self.out_names):
            out_dict[name] = tensor
            out_dict[dbname] = tensor
        return out_dict

    def __call__(self, *, node_values=None, single_prediction=False, batch_size=None, **kwargs):
        """

        :param node_values: dict[node->tensor]
        :param single_prediction: (default false) if True, wrap a batch axis onto the input and off of the output
        :param batch_size: batch over groups of examples by splitting the inputs along the batch axis.
                           does all computation as one batch if batch_size=None
        :param kwargs: values for arguments, using either the input's name or dbname attribute
        :return: dictionary of output values. Contains db_name:value, name:value, and node:value pairs.
        """

        if node_values is not None:
            if len(kwargs) != 0:
                raise ValueError("Cannot call Predictor with both node_values and kwargs")
            pass
        else:
            node_values = {self.argmap[k]: v for k, v in kwargs.items()}

        if len(node_values) != len(self.inputs):
            raise ValueError(
                "Wrong number of node inputs. {} distinct inputs supplied, "
                " {} inputs required".format(len(node_values), len(self.inputs))
            )

        if single_prediction:
            node_values = {n: v.unsqueeze(0) for n, v in node_values.items()}

        if batch_size is not None:
            out = self.predict_batched(node_values, batch_size)
        else:
            out = self.predict_all(node_values)

        if single_prediction:
            out = {n: v.unsqueeze(0) for n, v in out.items()}

        out = self.wrap_outputs(out)

        return out

    def predict_batched(self, node_values, batch_size):

        # Split the input into batches
        node_values = {n: torch.split(t, batch_size) for n, t in node_values.items()}

        n_batches = arrdict_len(node_values)

        # Make a dictionary for each batch
        batched_inputs = [{n: t[i] for n, t in node_values.items()} for i in range(n_batches)]

        # Predict for each batch
        batched_outputs = [self.predict_all(b) for b in progress_bar(batched_inputs, desc=self.name, unit="batch")]

        # Concatenate the full set of outputs.
        out_keys = batched_outputs[0].keys()
        full_outputs = {k: torch.cat([batched_outputs[i][k] for i in range(n_batches)]) for k in out_keys}
        return full_outputs

    def apply_to_database(self, db, **kwargs):
        """
        Note: kwargs are passed to self.__call__, e.g. the ``batch_size`` parameter.
        """
        results = {}
        for split_name, split_arrdict in db.splits.items():
            input_names = [x.db_name for x in self.graph.input_nodes]
            dict_inputs = {k: split_arrdict[k] for k in input_names}
            results[split_name] = self(**dict_inputs, **kwargs)
        return results

    def predict_all(self, node_values):

        input_values = [node_values[n] for n in self.inputs]

        if self.model_device is not None:
            input_values = [inp.to(self.model_device) for inp in input_values]

        out_values = self.graph(*input_values)

        out_dict = {k: v for k, v in zip(self.outputs, out_values)}

        if not self.requires_grad:
            out_dict = {k: v.detach() for k, v in out_dict.items()}

        if self.return_device is not None:
            out_dict = {k: v.to(self.return_device) for k, v in out_dict.items()}

        return out_dict
