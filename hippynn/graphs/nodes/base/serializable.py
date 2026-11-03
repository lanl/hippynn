from typing import Optional


class SerializableNode:
    _unregistered_classes: list = [] # class-level registry
    serialization_attributes: Optional[tuple[str, ...]] = None # for child classes to override

    def to_record(self):
        if self.serialization_attributes:
            return {name: getattr(self, name) for name in self.serialization_attributes}
        else:
            return None

    def from_record(self):
        # Interface contract, make sure you implement this if subclassing!
        return NotImplemented
    
    def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            # Merge each ancestor's own declared serialization_attributes (in MRO order, base-first)
            serialization_attributes = []
            for base_cls in reversed(cls.__mro__):
                if hasattr(base_cls, "serialization_attributes"):
                    # Note! Nodes which are not serializable can still declare these attributes!
                    # That's why we check the Node class.
                    if base_cls.serialization_attributes:
                        serialization_attributes.extend(base_cls.serialization_attributes)
            cls.serialization_attributes = tuple(dict.fromkeys(serialization_attributes))
            SerializableNode._unregistered_classes.append(cls)

