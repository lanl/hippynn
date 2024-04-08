"""
Base nodes for sublcassing.
"""
from ... import indextypes
from .algebra import _CombNode, AtLeast2D

from .node_functions import _BaseNode


class Node(_CombNode):
    pass


class SingleNode(Node):
    pass


class InputNode(SingleNode):
    _input_names = ()
    """Node for getting information for the database."""
    requires_grad = False
    input_type_str = "Input"

    def __init__(self, name=None, db_name=None, index_state=None):

        if hasattr(self,"_index_state") and self._index_state is not None:
            if index_state is not None:
                if index_state != self._index_state:
                    raise ValueError(f"Cannot override IdxType {self._index_state} of node type {self.__class__.__name___} "
                                     f"with user-specified type {index_state}.")
        else:
            if index_state is not None:
                self._index_state = index_state

        if db_name is None and name is None:
            raise TypeError("Input node requires name or db_name arguments.")
        if name is None and db_name is not None:
            name = self.input_type_str + "(db_name='{}')".format(db_name)
        super().__init__(name=name, parents=(), db_name=db_name, module=None)


class LossInputNode(InputNode):
    """Node for getting information from the model (predicted) or database (true) into the loss."""

    def __init__(self, name, origin_node, db_name):
        super().__init__(name, db_name)
        self.origin_node = origin_node

    @property
    def pred(self):
        raise TypeError("Node {} of type {} already in loss graph".format(self, type(self)))

    @property
    def true(self):
        raise TypeError("Node {} of type {} already in loss graph".format(self, type(self)))


class LossPredNode(LossInputNode):
    def __init__(self, name, origin_node, db_name):
        super().__init__(name, origin_node, db_name)
        self._index_state = getattr(origin_node, "_index_state", indextypes.IdxType.NotFound)


class LossTrueNode(LossInputNode):
    def __init__(self, name, origin_node, db_name):
        super().__init__(name, origin_node, db_name)
        self._index_state = indextypes.reduce_funcs.db_state_of(origin_node._index_state)
        self._main_output = AtLeast2D((self,))

    @property
    def main_output(self):
        return self._main_output


_BaseNode._LossPredNode = LossPredNode
_BaseNode._LossTrueNode = LossTrueNode
