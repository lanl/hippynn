from . import SingleNode
from .serializable import _SerializableNode

from ...indextypes import IdxType
from ....layers import algebra as algebra_mods


class ValueNode(SingleNode, _SerializableNode):
    index_state = IdxType.Scalar  # By definition abstract values do not have a batch axis.
    serialization_attributes = ("value","_converted")

    def __init__(self, value, convert=True, module="auto"):
        name = "Value({})".format(str(value))
        self.value = value
        self._converted = convert
        if module == "auto":
            module = algebra_mods.ValueMod(value=value, convert=convert)
        super().__init__(name, (), module=module)

    @classmethod
    def from_record(cls, value, _converted, db_name, index_state, **kwargs):
        node = cls(value=value, convert=_converted)
        node.db_name = db_name
        node.index_state = index_state
        return node
