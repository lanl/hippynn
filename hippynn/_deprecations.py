"""
Developer tool for managing deprecation warnings.
"""
from . import settings
from itertools import chain
from .tools import SetStateMixin, GetAttrMixin
import warnings, contextlib, functools


class HippynnNameDeprecation(DeprecationWarning):
    def __init__(self, deprecations, old_module=None, new_module=None):

        deprecations = dict(deprecations)

        if not deprecations:
            raise ValueError("Arguments are required to be non-empty.")

        if old_module is not None:
            if not isinstance(old_module, str):
                old_module = old_module.__name__
            deprecations = {f"{old_module}.{k}": v for k, v in deprecations.items()}
        if new_module is not None:
            if not isinstance(new_module, str):
                new_module = new_module.__name__
            deprecations = {k: f"{new_module}.{v}" for k, v in deprecations.items()}

        for k, v in deprecations.copy().items():
            if not isinstance(v, str):
                try:
                    vname = v.__qualname__
                except:
                    vname = repr(v)
                deprecations[k] = f"{v.__module__}.{vname}"

        self.deprecations = deprecations

        message = ["The following hippynn names have been deprecated, use replacements:"]
        for k, v in deprecations.items():
            message.append(f"\t{k} -> {v}")
        message.append("Use HIPPYNN_DEPRECATION_WARNINGS=ignore to disable this warning.")

        message = "\n".join(message)

        super().__init__(message)

    @classmethod
    def from_single(cls, old, new, **kwargs):
        deprecations = {old: new}
        return cls(deprecations, **kwargs)

    @classmethod
    def from_list(cls, *args, **kwargs):

        new_items = {}
        for a in args:
            if isinstance(a, warnings.WarningMessage):
                if not isinstance(a.message, HippynnNameDeprecation):
                    rewarn(a)
                    continue
                else:
                    a = a.message
            new_items.update(a.deprecations)
        if not new_items:
            return None
        return cls(deprecations=new_items, **kwargs)


def rewarn(*message_list):
    for m in message_list:
        warnings.warn_explicit(m.message, m.category, m.filename, m.lineno)


@contextlib.contextmanager
def handle_deprecations(stacklevel=0):
    if settings.DEPRECATION_WARNINGS == "all":
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            yield

        new_warning = HippynnNameDeprecation.from_list(*recorded)
        if new_warning is not None:
            warnings.warn(new_warning, stacklevel=2 + stacklevel)

    elif settings.DEPRECATION_WARNINGS == "simple":
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            yield

        found_deprecations = False
        for m in recorded:
            if not isinstance(m.message, HippynnNameDeprecation):
                rewarn(m)
            else:
                found_deprecations = True
        if found_deprecations:
            warnings.warn(
                "Deprecated hippynn variable names were encountered. If you encounter this message while "
                "loading a checkpoint, you can disable this message by saving it back again or "
                "using HIPPYNN_DEPRECATION_WARNINGS=ignore ."
            )

    elif settings.DEPRECATION_WARNINGS == "ignore":
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("ignore", HippynnNameDeprecation)
            yield
    else:
        raise ValueError(f"unknown setting: {settings.DEPRECATION_WARNINGS}")

    return


def bundles_deprecated_warnings(wrapped=None, stacklevel=None):
    if wrapped is None:
        return functools.partial(bundles_deprecated_warnings, stacklevel=stacklevel)

    if stacklevel is None:
        stacklevel = 0

    @functools.wraps(wrapped)
    def inner(*args, **kwargs):
        with handle_deprecations(stacklevel=stacklevel):
            return wrapped(*args, **kwargs)

    return inner


def warn_name_change(old, new, old_module=None, new_module=None, stacklevel=2, **kwargs):
        """
        """
        stacklevel +=1
        warning = HippynnNameDeprecation.from_single(old, new, old_module=old_module, new_module=new_module)
        warnings.warn(warning, stacklevel=stacklevel, **kwargs)



class _DeprecatedNamesMixin(SetStateMixin, GetAttrMixin):
    _DEPRECATED_NAMES: dict[str, str] = {}
    _DEPRECATED_STATE: dict[str, str] = {}

    def __setattr__(self, name: str, value):
        if name in self._DEPRECATED_NAMES:
            new_name = self._DEPRECATED_NAMES[name]
            warn_name_change(name, new_name)
            name = new_name
        super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in self._DEPRECATED_NAMES:
            new_name = self._DEPRECATED_NAMES[name]
            warn_name_change(name, new_name)
            return getattr(self, new_name)
        return super().__getattr__(name)

    def __setstate__(self, state: dict):
        for old_name, new_name in chain(self._DEPRECATED_NAMES.items(), self._DEPRECATED_STATE.items()):
            if old_name in state:
                warn_name_change(old_name, new_name)
                state[new_name] = state.pop(old_name)
        super().__setstate__(state)

    def __init_subclass__(cls, **kwargs):

        # Here we collect the inherited set of deprecated names.
        _DEPRECATED_NAMES = {}
        _DEPRECATED_STATE = {}

        for base_cls in reversed(cls.__mro__):
            if issubclass(base_cls, _DeprecatedNamesMixin):
                _DEPRECATED_NAMES.update(base_cls._DEPRECATED_NAMES)
                _DEPRECATED_STATE.update(base_cls._DEPRECATED_STATE)


        cls._DEPRECATED_NAMES = _DEPRECATED_NAMES
        cls._DEPRECATED_STATE = _DEPRECATED_STATE

        super().__init_subclass__(**kwargs)
