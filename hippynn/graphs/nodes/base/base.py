"""
Base nodes for sublcassing.
"""
from ... import indextypes
from .algebra import _NodeAlgebra

from .node_functions import BaseNode
from .algebra import UnaryNode, BinNode, _AlgebraicNode
from ....layers import algebra as algebra_mods



### Base class and automatically generated nodes. 
class Node(BaseNode, _NodeAlgebra):
    pass

class ValueNode(Node):
    _index_state = indextypes.IdxType.Scalar  # By definition abstract values do not have a batch axis.
    def __init__(self, value, convert=True):
        name = "Value({})".format(str(value))
        self.value = value
        self._converted = convert
        super().__init__(name, parents=(), module="auto")

    def auto_module(self):
        return algebra_mods.ValueMod(self.value, convert=self._converted)


class InvNode(Node, UnaryNode, _AlgebraicNode, algebraic_operation="invert"):
    pass

class NegNode(Node, UnaryNode, _AlgebraicNode, algebraic_operation="neg"):
    pass

class AddNode(Node, BinNode, _AlgebraicNode, algebraic_operation="add"):
    pass

class SubNode(Node, BinNode, _AlgebraicNode, algebraic_operation="sub"):
    pass

class MulNode(Node, BinNode, _AlgebraicNode, algebraic_operation="mul"):
    pass


class DivNode(Node, BinNode, _AlgebraicNode, algebraic_operation="truediv"):
    pass


class PowNode(Node, BinNode, _AlgebraicNode, algebraic_operation="pow"):
    pass


# This Node exists to prevent potential broadcasting problems, for example in the loss.
# Model-based quantities all use a feature index, even if the size is 1,
# e.g. energy is predicted with shape (n_molecules, 1)
# This AtLeast2D is then used to wrap things coming from the database so that they will
# have at least two dimensions.
# See nodes/loss.py and turn on `debug_loss_broadcast` if you have concerns about
# broadcasting behavior.
class AtLeast2D(Node):
    torch_module = algebra_mods.AtLeast2D()
    _index_state = indextypes.IdxType.NotFound

    def __init__(self, parents, *args, **kwargs):
        if len(parents) != 1:
            raise ValueError("AtLeast2D can only have 1 parent, got {}".format(len(parents)))
        p = parents[0]
        self._index_state = p._index_state
        super().__init__("Atleast2D({})".format(p), parents, *args, module=None, **kwargs)
        self.origin_node = p.origin_node


### Classes for deriving from

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


BaseNode._LossPredNode = LossPredNode
BaseNode._LossTrueNode = LossTrueNode
