"""
Base nodes for sublcassing.
"""
from ... import indextypes
from ...indextypes import IdxType
from .algebra import _NodeAlgebra

from .node_functions import _NodeFunctions
from .serializable import _SerializableNode
from .algebra import UnaryNode, BinNode, _AlgebraicOperation
from ....layers import algebra as algebra_mods
from ...._deprecations import _DeprecatedNamesMixin

from typing import Optional


class Node(_NodeFunctions, _NodeAlgebra):
    serialization_attributes = ("db_name",)  # hack


class SingleNode(Node, _DeprecatedNamesMixin):
    index_state: Optional[IdxType] = IdxType.Unlabeled
    serialization_attributes = ("index_state",)
    _DEPRECATED_NAMES = {
        "_index_state" : "index_state",
        }
    _DEPRECATED_STATE = {
        "db_name" : "_db_name",
    }
    
    @property
    def pred(self):
        if self._pred is None:
            if self.is_in_loss_graph():
                raise TypeError("Node {} of type {} already in loss graph".format(self, type(self)))
            return LossPredNode(self.name + "-pred", origin_node=self) # sets backlink when created
        return self._pred

    @property
    def true(self):
        if self._true is None:
            if self.is_in_loss_graph():
                raise TypeError("Node {} of type {} already in loss graph".format(self, type(self)))
            return LossTrueNode(self.name + "-true", origin_node=self) # sets backlink when created
        return self._true
    
    @property
    def main_output(self):
        return self


class InvNode(UnaryNode, SingleNode, _AlgebraicOperation, operation="invert"):
    pass

class NegNode(UnaryNode, SingleNode,  _AlgebraicOperation, operation="neg"):
    pass

class AddNode(BinNode, SingleNode, _AlgebraicOperation, operation="add"):
    pass

class SubNode(BinNode, SingleNode, _AlgebraicOperation, operation="sub"):
    pass

class MulNode(BinNode, SingleNode, _AlgebraicOperation, operation="mul"):
    pass

class DivNode(BinNode, SingleNode, _AlgebraicOperation, operation="truediv"):
    pass

class PowNode(BinNode, SingleNode , _AlgebraicOperation, operation="pow"):
    pass


# This Node exists to prevent potential broadcasting problems, for example in the loss.
# Model-based quantities all use a feature index, even if the size is 1,
# e.g. energy is predicted with shape (n_systems, 1)
# This AtLeast2D is then used to wrap things coming from the database so that they will
# have at least two dimensions.
# See nodes/loss.py and turn on `debug_loss_broadcast` if you have concerns about
# broadcasting behavior.
class AtLeast2D(UnaryNode, SingleNode):
    torch_module = algebra_mods.AtLeast2D()
    _classname = "Atleast2D"

    def __init__(self, in_node):
        super().__init__(in_node)
        self.origin_node = in_node.origin_node


class InputNode(SingleNode, _SerializableNode):
    """Node for getting information for the database."""
    input_names = ()
    input_type_str = "Input"
    requires_grad: bool = False
    serialization_attributes = ("requires_grad",)

    def __init__(self, name=None, db_name=None, index_state=None):

        self.requires_grad = False

        # todo: just reduce signature on subclasses?
        if hasattr(self, "index_state") and self.index_state is not IdxType.Unlabeled:
            if index_state is not None:
                if index_state != self.index_state:
                    raise ValueError(f"Cannot override IdxType {self.index_state} of node type {self.__class__.__name__} "
                                     f"with user-specified type {index_state}.")
        else:
            if index_state is not None:
                self.index_state = index_state
        
        if db_name is None and name is None:
            raise TypeError("Input node requires name or db_name arguments.")
        if name is None and db_name is not None:
            name = db_name
        super().__init__(name=name, parents=(), db_name=db_name, module=None)


    @classmethod
    def from_record(cls, name, db_name, index_state, requires_grad, **kwargs):
        node = cls(name=name, db_name=db_name, index_state=index_state)
        if requires_grad:
            node.requires_grad = requires_grad
        return node


class LossInputNode(InputNode):
    """Node for getting information from the model (predicted) or database (true) into the loss."""
    serialization_attributes = ("origin_node",)

    def __init__(self, name, origin_node):
        super().__init__(name, db_name=None)
        self.origin_node = origin_node

    @classmethod
    def from_record(cls, name, origin_node, **kwargs):
        node = cls(name=name, origin_node=origin_node)
        # `LossPredNode`/`LossTrueNode` derive `index_state` from `origin_node` in their own
        # `__init__`, but a bare `LossInputNode` does not - restore it if it was independently set.
        if "index_state" in kwargs:
            node.index_state = kwargs["index_state"]
        return node


class LossPredNode(LossInputNode):
    def __init__(self, name, origin_node):
        super().__init__(name, origin_node)
        self.index_state = getattr(origin_node, "index_state", IdxType.Unlabeled)
        self.origin_node._pred = self


class LossTrueNode(LossInputNode):
    def __init__(self, name, origin_node):
        super().__init__(name, origin_node)
        self.index_state = indextypes.reduce_funcs.db_state_of(origin_node.index_state)
        self._main_output = AtLeast2D(self)
        self.origin_node._true = self

    @property
    def main_output(self):
        return self._main_output
    