"""
A base node that provides several output tensors.
"""
from ....layers.algebra import Idx
from .base import SingleNode, Node
from .. import _debprint
from ...indextypes import IdxType
from ...._deprecations import _DeprecatedNamesMixin


class IndexNode(SingleNode):
    input_names = ("parent",)

    def __init__(self, name, parents, index, index_state=None):
        if len(parents) != 1:
            raise TypeError("Index node takes exactly one parent.")
        par = parents[0]
        iname = par.output_names[index] if hasattr(par, "output_names") else "<{index}>".format(index=index)
        repr_info = {"parent_name": par.name, "index": iname}
        module = Idx(index, repr_info=repr_info)
        self.index = index
        self.index_state = IdxType.Unlabeled if index_state is None else index_state
        super().__init__(name, parents, module=module)
            

class MultiNode(Node,_DeprecatedNamesMixin):  # Multinode
    output_names = NotImplemented
    output_index_states = NotImplemented  # optional?
    main_output_name = NotImplemented
    _DEPRECATED_NAMES = {
    "_main_output" : "main_output_name",
    "_output_names": "output_names",
    "_output_index_states": "output_index_states"
}

    def __init__(self, name, parents, module="auto", *args, db_name=None, **kwargs):

        super().__init__(name, parents, *args, module=module, **kwargs)

        if self.output_index_states is NotImplemented:
            raise TypeError(f"no defined output index in {self.__class__.__name__}")
        
        if self.output_names is NotImplemented:
            raise TypeError(f"no defined output names in {self.__class__.__name__}")

        self.children = tuple(
            IndexNode(name + "." + cn, (self,), index=i, index_state=cidx)
            for i, (cn, cidx) in enumerate(zip(self.output_names, self.output_index_states))
        )
    
        if db_name is not None:
            self.db_name = db_name


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Enforce _child_index_states has same length as output_names
        if cls.output_index_states is not NotImplemented:
            if len(cls.output_index_states) != len(cls.output_names):
                raise AssertionError(
                    "Lengths of _child_index_states {} doesn't match lengths of ouput_names {}".format(
                        cls.output_index_states, cls.output_names
                    )
                )

        # Enforce no name conflict between input names and output names
        if cls.input_names is not NotImplemented:
            try:
                assert all(o not in cls.input_names for o in cls.output_names)
            except AssertionError as ae:
                raise ValueError(
                    "Multi-node output names {} conflict with input names {}".format(
                        cls.output_names, cls.input_names
                    )
                ) from ae
            
    @property
    def pred(self):
        return self.main_output.pred
    
    @property
    def true(self):
        return self.main_output.true
    
    @property
    def db_name(self):
        return None
        
    @db_name.setter
    def db_name(self, value):
        if value is not None:
            self.main_output.db_name = value

    def __dir__(self):
        dir_ = super().__dir__()
        if self.output_names is not NotImplemented:
            dir_ = dir_ + list(self.output_names)
        return dir_

    def __getattr__(self, item):

        if item in ("children", "output_names"):  # Guard against recursion
            raise AttributeError("Attribute {} not yet present.".format(item))        
        
        try:
            return self.children[self.output_names.index(item)]
        except (AttributeError, ValueError):
            pass

        return super().__getattr__(item)
            

    @property
    def main_output(self):
        if self.main_output_name is NotImplemented:
            raise TypeError(f"Main output not implemented for node type: {type(self)!r}")
        return getattr(self, self.main_output_name)

