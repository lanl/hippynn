"""
Package for tracking node index state.
The index state describes the meaning of the
batch axis or axes for the tensor. i.e. system level, atom level, pair level.
"""
import warnings

from ... import settings

if settings.DEBUG_AUTOINDEXING:
    warnings.warn("Printing automatic index coercion info! Output will be verbose.")


def _debprint(*args, **kwargs):
    if settings.DEBUG_AUTOINDEXING:
        print("AutoIndex:", *args, **kwargs)
    else:
        pass


from .type_def import IdxType
from .registry import register_index_transformer, clear_index_cache
from .reduce_funcs import (
    elementwise_compare_reduce,
    get_reduced_index_state,
    index_type_coercion,
    soft_index_type_coercion,
    db_form,
)

__all__ = [
    "IdxType",
    "register_index_transformer",
    "clear_index_cache",
    "elementwise_compare_reduce",
    "get_reduced_index_state",
    "index_type_coercion",
    "soft_index_type_coercion",
    "db_form",
]
