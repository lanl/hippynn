"""
Definitions of nodes for graph computation.
"""
import warnings

from ... import settings

if settings.DEBUG_NODE_CREATION:
    warnings.warn("Printing automatic node creation info! Output will be verbose.")


def _debprint(*args, **kwargs):
    if settings.DEBUG_NODE_CREATION:
        print("AutoNode:", *args, **kwargs)
    else:
        pass


from . import base, inputs, indexers, networks, targets, physics, loss
