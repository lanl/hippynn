"""
Machinery for nodes to support native python operators
such as addition, multiplication, subtraction.
"""
import functools
import operator

from ....layers import algebra as algebra_mods


from ...indextypes import elementwise_compare_reduce
from .node_functions import BaseNode


def wrap_as_node(obj):
    from .base import ValueNode
    return obj.main_output if isinstance(obj, BaseNode) else ValueNode(obj)


def coerces_values_to_nodes(func):
    """Wraps non-nodes as ValueNodes."""

    @functools.wraps(func)
    def newfunc(*args):
        return func(*(wrap_as_node(a) for a in args))

    return newfunc

BINARY_OPS_SUPPORTED = { "add", "sub", "mul", "truediv", "pow"}
REV_BINARY_OPS_SUPPORTED = {'r'+op for op in BINARY_OPS_SUPPORTED}
UNARY_OPS_SUPPORTED = {"invert", "neg"}
ALL_OPS_SUPPORTED = BINARY_OPS_SUPPORTED | REV_BINARY_OPS_SUPPORTED | UNARY_OPS_SUPPORTED
OPS_REMAINING = ALL_OPS_SUPPORTED.copy()

class _NodeAlgebra():
    """Inherit from this to get access to registered algebraic ops."""
    @classmethod
    def register_operation(cls, op_name, register_function):
        try:
            OPS_REMAINING.remove(op_name)
        except KeyError:
            raise TypeError(f"Operator {op_name!r} has already been registered!")
        full_name = "__" + op_name + "__"
        setattr(_NodeAlgebra, full_name, register_function)
        return 

class _AlgebraicNode():
    """Inherit from this to register an algebraic operation with keyword argument algebraic_operation."""
    def __init_subclass__(cls, *args, algebraic_operation, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        if algebraic_operation not in ALL_OPS_SUPPORTED:
            raise TypeError(f"Operator {op_name!r} not supported!")

        base_function = getattr(operator, algebraic_operation)
        register_function = coerces_values_to_nodes(base_function)
        _NodeAlgebra.register_operation(algebraic_operation, register_function)

        if algebraic_operation in BINARY_OPS_SUPPORTED:
            
            @functools.wraps(base_function)
            @coerces_values_to_nodes
            def register_function(self, other):
                return function(other, self)

            _NodeAlgebra.register_operation('r' + algebraic_operation, register_function)

        cls.torch_module = algebra_mods.LambdaModule(base_function)
        cls._classname = algebraic_operation
        return 


class UnaryNode():
    def __init__(self, in_node):
        #name = "{}({})".format(self._classname, in_node)
        super().__init__(self._classname, (in_node,), module=None)
        self._index_state = in_node._index_state


class BinNode():
    def __init__(self, left, right):
        left, right = left.main_output, right.main_output
        left, right = elementwise_compare_reduce(left, right)
        #name = "{}({}, {})".format(self._classname, left.name, right.name)
        super().__init__(self._classname, (left, right), module=None)
        self._index_state = left._index_state

def __getattr__(name: str):
    import warnings
    if name.endswith("Node") or name == "AtLeast2D":
        # Backwards compatibility for unpickling prior models
        warnings.warn(
            f"{name!r} is a deprecated class name, and has likely been relocated to base.py." + \
            "If you encounter this warning while loading a model, you can re-serialize it to disable the warning.",
            DeprecationWarning,
            stacklevel=2,
        )
        from . import base
        return getattr(base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
