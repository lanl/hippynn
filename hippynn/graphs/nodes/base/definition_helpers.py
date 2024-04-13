"""
Tools for organizing node definitions.

Usage of these tools is optional; they support node definitions
but are not required.

.. Note::
   The functions in this module are intended purely for defining new types
   of nodes. They are not strictly necessary; they only aid in defining more
   complex node behavior in a simple fashion.
"""
import functools
import contextlib
from .. import _debprint

from . import _BaseNode
from ...indextypes import index_type_coercion, elementwise_compare_reduce, get_reduced_index_state


class AutoNoKw:
    _auto_module_class = NotImplemented

    def auto_module(self):
        return self._auto_module_class()


class AutoKw:
    _auto_module_class = NotImplemented

    def auto_module(self):
        kw = self.module_kwargs or {}  # Default to empty dictionary if Falsey
        return self._auto_module_class(**kw)


@contextlib.contextmanager
def temporary_parents(child, parents):
    """
    Context manager for temporarily connecting a node to a set of parents.
    This is used during parent expansion so that `find_relatives` and
    `find_unique_relatives` can treat the nodes as connected even though
    they are not fully formed.

    :param child:
    :param parents:
    :return: None
    """
    # Raise error if this is called on an already-build child.
    # (This function could be refactored to deal with this case.)
    assert not (
        hasattr(child, "parents") or hasattr(child, "children")
    ), "Temporary connection to node requires that it is not initialized."

    parset = set(parents)  # In case a node has the same parent twice.

    try:
        for p in parset:
            p.children = (*p.children, child)
        child.parents = parents
        child.children = ()
        yield
    finally:
        for p in parset:
            p.children = tuple(c for c in p.children if c is not child)
        del child.parents
        del child.children


class TupleTypeMismatch(Exception):
    pass


class AlwaysMatch:
    pass


def format_form_name(form):
    try:
        iter(form)
        formname = ", ".join(tuple(cls.__name__ for cls in form))
    except TypeError:
        formname = form.__name__
    return f"({formname})"


def _assert_tupleform(input_tuple, type_tuple):
    _debprint("ASSERTING FORM: ", input_tuple, type_tuple)

    # If it is a type of base node: return if it is the right type, or error.
    if isinstance(type_tuple, type):

        if type_tuple is AlwaysMatch:
            return True

        if isinstance(input_tuple, type_tuple):
            return True
        else:
            raise TupleTypeMismatch("Parent node did not match required type: {}".format(type_tuple))

    # If not, it must at least have the same length
    if not len(input_tuple) == len(type_tuple):
        raise TupleTypeMismatch(
            "Wrong length. {}!={}".format(len(input_tuple), len(type_tuple))
            + " \nInput: {} \nExpected: {}".format(input_tuple, type_tuple)
        )

    # If it does, it needs to have the correct type for each entry in the tuple.
    for i, (n, cls) in enumerate(zip(input_tuple, type_tuple)):
        if not isinstance(n, cls):
            raise TupleTypeMismatch(str(type_tuple) + ": wrong type at index {}".format(i))
    return True


# method wrapper for form storage
def adds_to_forms(fn):
    @functools.wraps(fn)
    def inner(self, *args, **kwargs):
        form = fn(self, *args, **kwargs)
        if self.matches is NotImplemented:
            self.matches = ()  # empty tuple
        self.matches = *self.matches, form

    return inner


class ParentExpander:
    """
    Manager object to register and implement
    optional steps in building a graph node.
    """

    def __init__(self):
        # Matches is a tuple of tuples of (form, matching function)
        self.matches = NotImplemented

    def __iter__(self):
        return iter(self.matches)

    def match(self, *form):
        """Decorator: the applied function will be a generic FormTransformer"""

        def inner(matchfn):
            if self.matches is NotImplemented:
                self.matches = ()  # empty tuple
            self.matches = *self.matches, FormTransformer(form, matchfn)
            return matchfn

        return inner

    def matchlen(self, length):
        """
        Decorator: The decorated function will be applied if the number
        of parents matches the given length.
        :param length:
        :return:
        """
        return self.match(*((_BaseNode,) * length))

    @adds_to_forms
    def assertion(self, *form):
        """
        Assert that the parents match a given form.

        :param form:
        :return:
        """
        return FormAssertion(form)

    @adds_to_forms
    def assertlen(self, length):
        """
        Assert that there are a given number of parents.

        :param length:
        :return:

        .. Note::
           It is recommended only to use this function once as the final stage
           of expanding a node's parents, to ensure that a node can be constructed
           directly from a satisfactory set of parents that doesn't require
           any expansion.
        """
        return FormAssertLength(length)

    @adds_to_forms
    def get_main_outputs(self):
        """
        Return the main outputs for any multinodes in the parents.

        :return:
        """
        return MainOutputTransformer(AlwaysMatch)

    @adds_to_forms
    def matched_idx_coercion(self, form, needed_index_states):
        """
        Apply coercion to the needed index states if the given form is present.
        :param form:
        :param needed_index_states:
        :return:
        """
        return IndexFormTransformer(form, needed_index_states)

    @adds_to_forms
    def require_compatible_idx_states(self):
        """
        Ensure that all parents have commensurate index states.
        :return:
        """
        return CompatibleIdxTypeTransformer(AlwaysMatch)

    @adds_to_forms
    def require_idx_states(self, *needed_index_states):
        """
        Always coerce the nodes into a needed index state.

        :param needed_index_states:
        :return:

        .. Note::
           It is recommended only to use this function once as the final stage
           of expanding a node's parents, to ensure that a node can be constructed
           directly from a satisfactory set of parents that doesn't require
           any expansion.
        """
        return IndexFormTransformer(AlwaysMatch, needed_index_states)

    def _merge(self, *bases):
        """
        Used for merging

        :param bases: base classes to merge with (from MRO)
        :return: None
        """

        *true_bases, cls = reversed(bases)

        relevant_classes = tuple(
            sup_class
            for sup_class in true_bases
            if issubclass(sup_class, ExpandParents) and sup_class is not ExpandParents
        )

        base_matches = tuple(form for sup_class in relevant_classes for form in sup_class._parent_expander.matches)

        _debprint("Bases found for merging:", relevant_classes)

        if len(base_matches):
            # Filter only to contain the earliest copy of each form.
            if self.matches is not NotImplemented:
                base_matches = *base_matches, *self.matches
            base_form_set = set()
            base_match_list = []
            for match in base_matches:
                if match not in base_form_set:
                    _debprint("Adding", match)
                    base_form_set.add(match)
                    base_match_list.append(match)
                else:
                    _debprint("Already applied", match)
            base_matches = base_match_list
            _debprint("Forms found for merging:", base_matches)

            self.matches = tuple(base_matches)
        else:

            _debprint("No matches found for merging!")


class FormHandler:
    def add_class_doc(self):
        return None

    pass


def _fn_doc_adjust(form, fn):
    if fn.__module__ is __name__:
        # Don't document functions that live here, they are used many many times.
        return
    try:
        existing = fn.__doc__ or ""
        lines = existing.split("\n")
        # Count spaces in each non-blank line
        spaces = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip() != ""]
        # Strip those with no spaces
        spaces = [x for x in spaces if x != 0]
        # Grab the min, if non exist, use 0
        spaces = min(spaces) if spaces else 0
        spaces = " " * spaces
        formname = format_form_name(form)
        fn.__doc__ = existing + f"\n{spaces}Used for creation from parents with signature {formname}\n{spaces}"

    except AttributeError:
        # Can't modify doc of a method dynamically.
        # This occurs when calling one of the explicit form handlers such as
        # AssertTypes or AssertLen
        pass


class FormTransformer(FormHandler):
    def __init__(self, form, fn):
        self.form = form
        self.fn = fn
        _fn_doc_adjust(form, fn)

    def add_class_doc(self):
        form_name = format_form_name(self.form)
        return f"If matching {form_name}, then apply " + (self.fn.__name__ or "")

    def __repr__(self):
        return f"ParentTransformer(sig={format_form_name(self.form)},fn={self.fn})"

    def __call__(self, node_self, *parents, purpose=None, **kwargs):
        try:
            _assert_tupleform(parents, self.form)
            _debprint("Expanding form", self.form)
            _debprint("Input parents:", self.form)
            _debprint("Match function:", self.fn)
            if purpose is None:
                purpose = "{}: Expanding parents {} based on form {}".format(type(self), parents, self.form)
            with temporary_parents(node_self, parents):
                # We have to pass self here explicitly because the form handler stores unbound functions.
                new_parents = self.fn(node_self, *parents, purpose=purpose, **kwargs)
            return new_parents
        except TupleTypeMismatch:
            _debprint("Didn't pass form!")
            return parents
        except Exception as ee:
            raise RuntimeError("Error while transforming {}".format(self.fn.__qualname__)) from ee


class IndexFormTransformer(FormTransformer):
    def __init__(self, form, idxstates):
        super().__init__(form, self.fn)
        self.idxstates = idxstates

    def add_class_doc(self):
        return f"Transforms the parents to have index states {self.idxstates}"

    def __repr__(self):
        return f"IndexTransformer from {format_form_name(self.form)} to {self.idxstates}"

    def fn(self, node_self, *parents, **kwargs):
        """Coerces index states for all parents"""
        new_parents = []
        for node, idxstate in zip(parents, self.idxstates):
            if idxstate is None:
                new = node
            else:
                new = index_type_coercion(node, idxstate)
            new_parents.append(new)
        return tuple(new_parents)


class MainOutputTransformer(FormTransformer):
    def __init__(self, form):
        super().__init__(form, self.fn)

    def add_class_doc(self):
        return """Gets main_output of nodes: casts MultiNodes to their main output"""

    @staticmethod
    def fn(node_self, *parents, **kwargs):
        """
        Gets main_output of nodes: casts multinodes to single nodes
        """
        return tuple(p.main_output for p in parents)


class CompatibleIdxTypeTransformer(FormTransformer):
    def __init__(self, form):
        super().__init__(form, self.fn)

    def add_class_doc(self):
        return """Attempts coercion of all inputs to the same index state."""

    @staticmethod
    def fn(node_self, *parents, **kwargs):
        """
        Enforces that all parents have compatible index states.
        """
        index_state = get_reduced_index_state(*parents)
        return parents


class FormAssertion(FormHandler):
    def __init__(self, form):
        self.form = form
        _debprint("Assertion created:", self)

    def add_class_doc(self):
        form_name = format_form_name(self.form)
        return f"Asserts that the parents have the form {form_name}."

    def __call__(self, self_cls, *parents, **kwargs):
        _debprint("Attempting Assertion", self)
        try:
            _assert_tupleform(parents, self.form)
        except TupleTypeMismatch as ee:
            raise TypeError("Input does not meet required form: \nInput: {} \nForm: {}".format(parents, self)) from ee
        return parents

    def __repr__(self):
        return super().__repr__() + "({})".format(self.form)


class FormAssertLength(FormAssertion):
    def __init__(self, length):
        self.length = length
        super().__init__((_BaseNode,) * length)

    def add_class_doc(self):
        return f"Asserts that the number of parents is {self.length}"

    def __repr__(self):
        return "FormLengthAssertion({})".format(self.length)


# This metaclass inserts the _parent_expander attribute into
# any class that has this metaclass.
# Note for developers: The metaclass is needed because
# the class must be modified before the class definition
# is executed; __init_subclass__ is run /after/ the class
# definition is executed.
class ExpandParentMeta(type):
    @classmethod
    def __prepare__(mcl, name, bases, **kwargs):
        cls_dict = super(ExpandParentMeta, mcl).__prepare__(name, bases, **kwargs)
        cls_dict["_parent_expander"] = ParentExpander()
        return cls_dict


def _append_docs(cls):
    """
    Add a note to documentation on parent expansion process.

    .. Note::
       This function assumes the class is defined at module-level scope.
       If not the indentation may be funny.
    """
    new_doc = (cls.__doc__ + "\n") if cls.__doc__ else ""
    new_doc += """\n    .. Note::\n       This node has parent expansion, following these procedures.\n\n"""
    for form_handler in cls._parent_expander:
        add = form_handler.add_class_doc()
        if add:
            new_doc += f"       #. {add}\n"
        else:
            raise ValueError("No documentation for transformation!")
    new_doc += "\n"
    return new_doc


class ExpandParents(metaclass=ExpandParentMeta):
    _parent_expander = None

    def __init_subclass__(cls, **kwargs):
        """
        Takes care of parent expansion setup and documentation for expansion of parents.
        """
        super().__init_subclass__(**kwargs)
        _debprint("Constructing parent expansion for ", cls)
        #  Note: Forms are applied in opposite order to MRO.
        #  This makes sense because initialization typically
        #  begins with the base class and then is customized or added to
        #  by a derived class.
        cls._parent_expander._merge(*cls.__mro__)

        # If no documentation, do not apply any decorating to the documentation
        if cls.__doc__:
            cls.__doc__ = _append_docs(cls)

    def expand_parents(self, parents, *, purpose=None, **kwargs):
        if isinstance(parents, _BaseNode):
            parents = (parents,)
        for form_handler in self._parent_expander:
            _debprint("Processing form:")
            _debprint("\tForm:", form_handler.form)
            _debprint("\t\targs:", parents)
            _debprint("\t\tkwargs:", kwargs)
            try:
                # We have to pass self here explicitly because the form handler stores unbound functions,
                # rather than bound methods.
                parents = form_handler(self, *parents, purpose=purpose, **kwargs)
            except Exception as ee:
                raise RuntimeError("Couldn't build {} automatically".format(type(self))) from ee
        return parents
