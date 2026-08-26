from . import SingleNode
from .node_functions import _SerializableNode

from ...indextypes import IdxType
from ....layers import algebra as algebra_mods


class ValueNode(SingleNode, _SerializableNode):
    index_state = IdxType.Scalar  # By definition abstract values do not have a batch axis.

    def __init__(self, value, convert=True, module="auto"):
        name = "Value({})".format(str(value))
        self.value = value
        self._converted = convert
        self.module_kwargs = {"value": value, "convert": convert}
        if module == "auto":
            module = algebra_mods.ValueMod(value=value, convert=convert)
        super().__init__(name, (), module=module)

    # `to_record` is attached generically in hippynn/serializer/hooks.py.

    @staticmethod
    def from_record(data, typ, *, ctx):
        """
        cattrs structure hook: reconstructs a ValueNode directly from its recorded
        ``value``/``convert`` module kwargs. ValueNode's constructor takes the value itself
        rather than following the ``(name, parents, ...)`` convention AutoModule.from_record
        assumes, so it needs its own override rather than the generic one.
        """
        name, parents, kwargs, record = data
        return typ(kwargs["value"], convert=kwargs["convert"])
