"""
A base node that provides several output tensors.
"""
from ....layers.algebra import Idx
from .base import SingleNode, Node
from .. import _debprint
from ...indextypes import IdxType


class IndexNode(SingleNode):
    _input_names = ("parent",)

    def __init__(self, name, parents, index, index_state=None):
        if len(parents) != 1:
            raise TypeError("Index node takes exactly one parent.")
        par = parents[0]
        iname = par._output_names[index] if hasattr(par, "_output_names") else "<{index}>".format(index=index)
        repr_info = {"parent_name": par.name, "index": iname}
        module = Idx(index, repr_info=repr_info)
        self.index = index
        self._index_state = IdxType.NotFound if index_state is None else index_state
        super().__init__(name, parents, module=module)


class MultiNode(Node):  # Multinode
    _output_names = NotImplemented
    _output_index_states = NotImplemented  # optional?
    _main_output = NotImplemented

    def __init__(self, name, parents, module="auto", *args, db_name=None, **kwargs):

        super().__init__(name, parents, *args, module=module, **kwargs)

        self.children = tuple(
            IndexNode(name + "." + cn, (self,), index=i, index_state=cidx)
            for i, (cn, cidx) in enumerate(zip(self._output_names, self._output_index_states))
        )
        self.main_output.db_name = db_name

    def set_dbname(self, db_name):
        self.main_output.set_dbname(db_name)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Enforce _child_index_states has same length as _output_names
        if cls._output_index_states is not NotImplemented:
            if len(cls._output_index_states) != len(cls._output_names):
                raise AssertionError(
                    "Lengths of _child_index_states {} doesn't match lengths of ouput_names {}".format(
                        cls._output_index_states, cls._output_names
                    )
                )

        # Enforce no name conflict between input names and output names
        if cls._input_names is not NotImplemented:
            try:
                assert all(o not in cls._input_names for o in cls._output_names)
            except AssertionError as ae:
                raise ValueError(
                    "Multi-node output names {} conflict with input names {}".format(
                        cls._output_names, cls._input_names
                    )
                ) from ae

    def __dir__(self):
        dir_ = super().__dir__()
        if self._output_names is not NotImplemented:
            dir_ = dir_ + list(self._output_names)
        return dir_

    def __getattr__(self, item):
        if item in ("children", "_output_names"):  # Guard against recursion
            raise AttributeError("Attribute {} not yet present.".format(item))
        try:
            return super().__getattr__(item)  # Defer to BaseNode first
        except AttributeError:
            pass
        try:
            return self.children[self._output_names.index(item)]
        except (AttributeError, ValueError):
            raise AttributeError("{} object has no attribute '{}'".format(self.__class__, item))

    @property
    def main_output(self):
        if self._main_output is NotImplemented:
            return super().main_output
        return getattr(self, self._main_output)
