"""
Serialization of hippynn Node graphs to/from plain data.

`types.py` defines the plain-data record shapes, `registry.py` the node-class registry and
per-class reconstruction dispatch, `processor.py` the cattrs-driven `Processor`, `hooks.py`
the tensor/node hooks `Processor` uses, and `api.py` the public `serialize`/`deserialize`
entry points (thin wrappers around `Processor.unstructure`/`structure`) plus the file-facing
`savegraph`/`loadgraph`. This module re-exports the public surface.
"""

from .types import PROTOCOL_VERSION, NodeInfo
from .registry import ensure_registry_consistency, register_structure_hooks
from .api import serialize, deserialize, savegraph, loadgraph
from . import hooks  # Trigger hook installation -- needed!

ensure_registry_consistency(raise_=False)
register_structure_hooks()
