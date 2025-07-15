"""
Base nodes for sublcassing.
"""
from ... import indextypes
from ...indextypes import IdxType
from .algebra import _NodeAlgebra

from .node_functions import _NodeFunctions
from .algebra import UnaryNode, BinNode, _AlgebraicOperation
from ....layers import algebra as algebra_mods
from ...._deprecations import _DeprecatedNamesMixin

from typing import Optional


class Node(_NodeFunctions, _NodeAlgebra):
    pass



class SingleNode(Node, _DeprecatedNamesMixin):
    index_state: Optional[IdxType] = IdxType.Unlabeled
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
            self._pred = LossPredNode(self.name + "-pred", origin_node=self, db_name=self.db_name)
        return self._pred

    @property
    def true(self):
        if self._true is None:
            if self.is_in_loss_graph():
                raise TypeError("Node {} of type {} already in loss graph".format(self, type(self)))
            self._true = LossTrueNode(self.name + "-true", origin_node=self, db_name=self.db_name)
        return self._true
    
    @property
    def db_name(self):
        return self._db_name

    @db_name.setter
    def db_name(self, value):
        self._db_name = value
        if not self.is_in_loss_graph():
            self.true.db_name = value
            self.pred.db_name = value
    
    @property
    def main_output(self):
        return self
    


class ValueNode(SingleNode):
    index_state = IdxType.Scalar  # By definition abstract values do not have a batch axis.
    def __init__(self, value, convert=True):
        name = "Value({})".format(str(value))
        self.value = value
        self._converted = convert
        module = algebra_mods.ValueMod(self.value, convert=self._converted)
        super().__init__(name, parents=(), module=module)


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
class AtLeast2D(SingleNode):
    torch_module = algebra_mods.AtLeast2D()
    index_state = IdxType.Unlabeled

    def __init__(self, parents, *args, **kwargs):
        if len(parents) != 1:
            raise ValueError("AtLeast2D can only have 1 parent, got {}".format(len(parents)))
        p = parents[0]
        self.index_state = p.index_state
        super().__init__("Atleast2D({})".format(p), parents, *args, module=None, **kwargs)
        self.origin_node = p.origin_node


class InputNode(SingleNode):
    input_names = ()
    """Node for getting information for the database."""
    requires_grad = False
    input_type_str = "Input"

    def __init__(self, name=None, db_name=None, index_state=None):

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
            name = self.input_type_str + "(db_name='{}')".format(db_name)
        super().__init__(name=name, parents=(), db_name=db_name, module=None)



class LossInputNode(InputNode):
    """Node for getting information from the model (predicted) or database (true) into the loss."""

    def __init__(self, name, origin_node, db_name):
        super().__init__(name, db_name)
        self.origin_node = origin_node


class LossPredNode(LossInputNode):
    def __init__(self, name, origin_node, db_name):
        super().__init__(name, origin_node, db_name)
        self.index_state = getattr(origin_node, "index_state", IdxType.Unlabeled)


class LossTrueNode(LossInputNode):
    def __init__(self, name, origin_node, db_name):
        super().__init__(name, origin_node, db_name)
        self.index_state = indextypes.reduce_funcs.db_state_of(origin_node.index_state)
        self._main_output = AtLeast2D((self,))

    @property
    def main_output(self):
        return self._main_output
    