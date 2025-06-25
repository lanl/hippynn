"""
Machinery for nodes to support native python operators
such as addition, multiplication, subtraction.
"""
import functools
import operator

from ....layers import algebra as algebra_mods


from ...indextypes import elementwise_compare_reduce


def wrap_as_node(obj):
    from .base import ValueNode, Node
    return obj.main_output if isinstance(obj, Node) else ValueNode(obj)


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
    @staticmethod
    def register_operation(operation, cls):
        try:
            OPS_REMAINING.remove(operation)
        except KeyError:
            raise TypeError(f"Operator {operation!r} has already been registered!")
        full_name = "__" + operation + "__"
        method = _make_node_method(operation, cls)
        setattr(_NodeAlgebra, full_name, method)
        return

def _make_node_method(operation, cls):

    if operation in BINARY_OPS_SUPPORTED:
        def method(self, other):
            return cls(self, other)
    elif operation in REV_BINARY_OPS_SUPPORTED:
        def method(self, other):
            return cls(other, self)
    elif operation in UNARY_OPS_SUPPORTED:
        def method(self):
            return cls(self)
        
    return coerces_values_to_nodes(method)

    
class _AlgebraicOperation():
    """Inherit from this to register an algebraic operation with keyword argument algebraic_operation."""
    def __init_subclass__(cls, *args, operation, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        if operation not in ALL_OPS_SUPPORTED:
            raise TypeError(f"Operator {op_name!r} not supported or already registered!")

        _NodeAlgebra.register_operation(operation, cls)
        base_function = getattr(operator, operation)

        if operation in BINARY_OPS_SUPPORTED:
            _NodeAlgebra.register_operation('r' + operation, cls)

        cls.torch_module = algebra_mods.LambdaModule(base_function)
        cls._classname = operation
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
    """
    Module-level getattr for finding of old functions.
    """
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
