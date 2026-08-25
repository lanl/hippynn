from . import SingleNode, AutoModule

from ...indextypes import IdxType
from ....layers import algebra as algebra_mods


class ValueNode(AutoModule, SingleNode):
    index_state = IdxType.Scalar  # By definition abstract values do not have a batch axis.
    auto_module_class = algebra_mods.ValueMod
    auto_module_kwargs = ("value", "convert")

    def __init__(self, value, convert=True, module="auto"):
        name = "Value({})".format(str(value))
        self.value = value
        self._converted = convert
        super().__init__(name, (), module=module, value=value, convert=convert)
