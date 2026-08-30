
# Todo rename to SerializableNode; it is not going to be hidden.
class _SerializableNode():
    _unregistered_classes: list = [] # class-level registry
    serialization_attributes: tuple[str, ...] = () # for child classes to override


    
    def to_record(self):
        from ....serializer import NodeInfo
        record = NodeInfo(parents=self.parents,
                        db_name=self.db_name)
        if self.serialization_attributes:
            record.extra_state = {name: getattr(self, name) for name in self.serialization_attributes}
        return record

    def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            # Merge each ancestor's own declared serialization_attributes (in MRO order, base-first)
            # into cls's own attribute, mirroring _DeprecatedNamesMixin.__init_subclass__.
            merged = []
            for base_cls in reversed(cls.__mro__):
                for name in base_cls.__dict__.get("serialization_attributes", ()):
                    if name not in merged:
                        merged.append(name)
            cls.serialization_attributes = tuple(merged)
            _SerializableNode._unregistered_classes.append(cls)

