"""
Machinery for nodes to support native python operators
such as addition, multiplication, subtraction.
"""
import functools
import operator

from ...indextypes import IdxType, get_reduced_index_state, elementwise_compare_reduce
from .node_functions import _BaseNode
from ....layers import algebra as algebra_mods


def wrap_as_node(obj):
    return obj.main_output if isinstance(obj, _BaseNode) else ValueNode(obj)


def coerces_values_to_nodes(func):
    """Wraps non-nodes as ValueNodes."""

    @functools.wraps(func)
    def newfunc(*args):
        return func(*(wrap_as_node(a) for a in args))

    return newfunc


class _CombNode(_BaseNode):
    @coerces_values_to_nodes
    def __add__(self, other):
        return AddNode(self, other)

    @coerces_values_to_nodes
    def __sub__(self, other):
        return SubNode(self, other)

    @coerces_values_to_nodes
    def __mul__(self, other):
        return MulNode(self, other)

    @coerces_values_to_nodes
    def __truediv__(self, other):
        return DivNode(self, other)

    @coerces_values_to_nodes
    def __pow__(self, other):
        return PowNode(self, other)

    @coerces_values_to_nodes
    def __radd__(self, other):
        return AddNode(other, self)

    @coerces_values_to_nodes
    def __rsub__(self, other):
        return SubNode(other, self)

    @coerces_values_to_nodes
    def __rmul__(self, other):
        return MulNode(other, self)

    @coerces_values_to_nodes
    def __rtruediv__(self, other):
        return DivNode(other, self)

    @coerces_values_to_nodes
    def __rpow__(self, other):
        return DivNode(other, self)

    def __invert__(self):
        return InvNode(self)

    def __pos__(self):
        return self

    def __neg__(self):
        return NegNode(self)


class ValueNode(_CombNode):
    _index_state = IdxType.Scalar

    def __init__(self, value, convert=True):
        name = "Value({})".format(str(value))
        self.value = value
        self._converted = convert
        super().__init__(name, parents=(), module="auto")

    def auto_module(self):
        return algebra_mods.ValueMod(self.value, convert=self._converted)


class _PredefinedOp:
    def __init_subclass__(cls, *, op=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if op is not None:
            cls.torch_module = algebra_mods.LambdaModule(op)
            cls._classname = op.__name__


class UnaryNode(_PredefinedOp, _CombNode):
    def __init__(self, in_node):
        name = "{}({})".format(self._classname, in_node)
        super().__init__(name, (in_node,), module=None)
        self._index_state = in_node._index_state


class InvNode(UnaryNode, op=operator.invert):
    pass


class NegNode(UnaryNode, op=operator.neg):
    pass


class BinNode(_PredefinedOp, _CombNode):
    _classname = None

    def __init__(self, left, right):
        left, right = left.main_output, right.main_output
        idxstate = get_reduced_index_state(left, right)
        left, right = elementwise_compare_reduce(left, right)
        name = "{}({}, {})".format(self._classname, left.name, right.name)
        super().__init__(name, (left, right), module=None)
        self._index_state = idxstate


class AddNode(BinNode, op=operator.add):
    pass


class SubNode(BinNode, op=operator.sub):
    pass


class MulNode(BinNode, op=operator.mul):
    pass


class DivNode(BinNode, op=operator.truediv):
    pass


class PowNode(BinNode, op=operator.pow):
    pass


# This Node exists to prevent potential broadcasting problems, for example in the loss.
# Model-based quantities all use a feature index, even if the size is 1,
# e.g. energy is predicted with shape (n_molecules, 1)
# This AtLeast2D is then used to wrap things coming from the database so that they will
# have at least two dimensions.
# See nodes/loss.py and turn on `debug_loss_broadcast` if you have concerns about
# broadcasting behavior.
class AtLeast2D(_BaseNode):
    torch_module = algebra_mods.AtLeast2D()
    _index_state = IdxType.NotFound

    def __init__(self, parents, *args, **kwargs):
        if len(parents) != 1:
            raise ValueError("AtLeast2D can only have 1 parent, got {}".format(len(parents)))
        p = parents[0]
        self._index_state = p._index_state
        super().__init__("Atleast2D({})".format(p), parents, *args, module=None, **kwargs)
        self.origin_node = p.origin_node
