from . import SingleNode
from .serializable import _SerializableNode

from ...indextypes import IdxType
from ....layers import algebra as algebra_mods


class ValueNode(SingleNode, _SerializableNode):
    index_state = IdxType.Scalar  # By definition abstract values do not have a batch axis.
    serialization_attributes = ("module_kwargs",)

    def __init__(self, value, convert=True, module="auto"):
        name = "Value({})".format(str(value))
        self.value = value
        self._converted = convert
        self.module_kwargs = {"value": value, "convert": convert}
        if module == "auto":
            module = algebra_mods.ValueMod(value=value, convert=convert)
        super().__init__(name, (), module=module)

    @staticmethod
    def from_record(name, info, typ, *, ctx):
        """
        cattrs structure hook: reconstructs a ValueNode directly from its recorded
        ``value``/``convert`` module kwargs. ValueNode's constructor takes the value itself
        rather than following the ``(name, parents, ...)`` convention AutoModule.from_record
        assumes, so it needs its own override rather than the generic one.
        """
        module_kwargs = info.extra_state["module_kwargs"]
        return typ(module_kwargs["value"], convert=module_kwargs["convert"])
