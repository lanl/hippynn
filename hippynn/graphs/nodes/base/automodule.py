from typing import Optional, Union, Tuple, Dict

from .node_functions import _SerializableNode

class AutoModule(_SerializableNode):
    """
    Helper mixin class for piping keyword arguments into an nn.module class.

    Using AutoModule is required for text-based graph serialization.

    Keyword Argument sources (lower precedence first):

        - module_kwargs argument to init.
        - kwarg-source pairs from self.auto_module_kwargs,
          where the value for the source is popped from the kwargs to the node.
        - self.module_kwargs.

    Note: If self.auto_module_kwargs is not a dict, it will then be interpreted
      as a list of keys for a dictionary with the identity for key-value mapping.

    After constructing the kwargs, they are saved as self.module_kwargs.

    """
    # developer note: dictionary form is a bit sensitive for node serialization, handle with care.
    # For this reason, an AutoModule node MUST allow passing of all auto_module_kwargs
    # in at the node level
    auto_module_kwargs: Optional[Union[Tuple[str], Dict[str, str]]] = None

    def __init__(self, *args, module="auto", module_kwargs=None, **kwargs):

        # if none supplied, start with empty
        if module_kwargs is None:
            module_kwargs = {}

        auto_module_kwargs = self.auto_module_kwargs

        # if automatic keyword translation is enabled,
        # set that up
        if auto_module_kwargs is None:
            auto_module_kwargs = {}
        elif not isinstance(auto_module_kwargs, dict):
            # Assume list of keys-values which are identical.
            auto_module_kwargs = dict(zip(auto_module_kwargs, auto_module_kwargs))

        # Pipe auto-forward keywords from node kwargs
        # into module creation kwargs
        for k, v in auto_module_kwargs.items():
            if v in kwargs:
                module_kwargs[k] = kwargs.pop(v)

        # Add any that have been saved via self.module_kwargs
        module_kwargs |= getattr(self, "module_kwargs", {})

        if module == "auto":
            self.module_kwargs = module_kwargs
            module = self.auto_module_class(**module_kwargs)
            self._auto_module_captured = True
        else:
            self._auto_module_captured = False

        super().__init__(*args, module=module, **kwargs)

    # `unstructure` is attached generically in hippynn/serializer/hooks.py.

    @staticmethod
    def structure(data, typ, *, ctx):
        """
        cattrs structure hook: reverses the translation performed by ``__init__``. Keys named
        in ``auto_module_kwargs`` are passed through as node-level keywords (translated to
        their node-level name), and everything else is bundled into ``module_kwargs``.
        """
        name, parents, kwargs, record = data
        auto_module_kwargs = typ.auto_module_kwargs
        if not auto_module_kwargs:
            mapping = {}
        elif isinstance(auto_module_kwargs, dict):
            mapping = auto_module_kwargs
        else:
            mapping = dict(zip(auto_module_kwargs, auto_module_kwargs))

        node_kwargs = {}
        leftover_module_kwargs = {}
        for k, v in kwargs.items():
            if k in mapping:
                node_kwargs[mapping[k]] = v
            else:
                leftover_module_kwargs[k] = v

        return typ(name, parents, module_kwargs=leftover_module_kwargs, **node_kwargs)
