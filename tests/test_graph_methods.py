import pytest

from collections.abc import Iterable

from conftest import energy_graph


def test_nodes_from_name_single(energy_graph):
    from hippynn.graphs.nodes.base import Node

    result = energy_graph.nodes_from_name("T")
    assert isinstance(result, Iterable)
    assert len(result) == 1
    assert isinstance(result[0], Node)


def test_nodes_from_name_multiple(energy_graph):
    from hippynn.graphs.nodes.base import Node

    result = energy_graph.nodes_from_name(None)
    assert isinstance(result, Iterable)
    assert len(result) > 1


def test_unique_node_from_name_pass(energy_graph):
    energy_graph.unique_node_from_name("T")


def test_unique_node_from_name_fail_notfound(energy_graph):
    from hippynn.graphs.nodes.base import NodeNotFound

    with pytest.raises(NodeNotFound):
        energy_graph.unique_node_from_name("cowabunga")


def test_unique_node_from_name_fail_multiple(energy_graph):
    from hippynn.graphs.nodes.base import NodeAmbiguityError

    with pytest.raises(NodeAmbiguityError):
        energy_graph.unique_node_from_name(None)


def test_node_from_name_depreciation(energy_graph):
    from hippynn._deprecations import HippynnNameDeprecation

    with pytest.warns(HippynnNameDeprecation):
        energy_graph.node_from_name("T")
