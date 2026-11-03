import pickle

import pytest

import hippynn
from hippynn.graphs.indextypes import IdxType

from conftest import skip_if_no_models, MODEL_DIR
from conftest import ignore_cusp_warning, ignore_relocation, ignore_weights_only_warning, ignore_sensitivity_warning
from conftest import ignore_registry_mismatch_warning
import torch

pytestmark = ignore_registry_mismatch_warning


@pytest.fixture
def processor_pair():
    """
    The (single, process-wide) writer/reader Processor, reset to a clean state. `writer`/`reader`
    are the same object - there is only one `context_converter` - but `.structure_with_context()`
    always resets internally before reading (see its docstring), so write and read remain
    independent.
    """
    from hippynn.serializer.processor import CONTEXT_CONVERTER

    CONTEXT_CONVERTER.reset()
    return CONTEXT_CONVERTER, CONTEXT_CONVERTER


PLAIN_DATA_CASES = [
    pytest.param(1, id="int"),
    pytest.param(2., id="float"),
    pytest.param("hello world", id="str"),
    pytest.param({"a": 1}, id="dict"),
    pytest.param([1, 2.5, "three", None, (4, 5)], id="list"),
    pytest.param({"a": [{'thing':(27,None)}], "b": {"nested": True}}, id="nested"),
    pytest.param({"idx": IdxType.Scalar}, id="indextype"),
]
@pytest.mark.parametrize("data", PLAIN_DATA_CASES)
def test_processor_plain_data_roundtrip(processor_pair, data):
    writer, reader = processor_pair

    structure, tensor_blob, _node_blob = writer.unstructure_with_context(data)
    assert tensor_blob == []

    restored = reader.structure_with_context(structure, tensor_blob=tensor_blob, node_blob=_node_blob)
    assert restored == data


def test_processor_tensor_roundtrip(processor_pair):
    writer, reader = processor_pair

    data = {"weight": torch.arange(6).reshape(2, 3)}

    structure, tensor_blob, _node_blob = writer.unstructure_with_context(data)
    assert len(tensor_blob) == 1

    restored = reader.structure_with_context(structure, tensor_blob=tensor_blob, node_blob=_node_blob)
    assert torch.equal(restored["weight"], data["weight"])


def test_processor_shared_tensor_roundtrip(processor_pair):
    writer, reader = processor_pair

    shared = torch.ones(3)
    data = {"a": shared, "b": shared}

    structure, tensor_blob, _node_blob = writer.unstructure_with_context(data)
    assert len(tensor_blob) == 1, "the same tensor object referenced twice must be stored once"

    restored = reader.structure_with_context(structure, tensor_blob=tensor_blob, node_blob=_node_blob)
    assert restored["a"] is restored["b"], "shared tensor identity must survive the round-trip"


def test_processor_set_roundtrip(processor_pair):
    writer, reader = processor_pair

    data = {"values": {1, 2, 3}}

    structure, tensor_blob, _node_blob = writer.unstructure_with_context(data)
    restored = reader.structure_with_context(structure, tensor_blob=tensor_blob, node_blob=_node_blob)

    assert restored["values"] == data["values"]


def test_processor_warns_tensor_count(processor_pair):
    writer, reader = processor_pair

    data = {"weight": torch.ones(3)}
    structure, tensor_blob, node_blob = writer.unstructure_with_context(data)

    tensor_blob = list(tensor_blob) + [torch.zeros(1)]

    with pytest.warns(UserWarning, match="tensor"):
        reader.structure_with_context(structure, tensor_blob=tensor_blob, node_blob=node_blob)


def test_processor_warns_node_count(processor_pair):
    from hippynn.graphs.nodes.base import ValueNode

    writer, reader = processor_pair

    node = ValueNode(1.0)
    structure, tensor_blob, node_blob = writer.unstructure_with_context(node)

    node_blob = list(node_blob)
    orphan_entry = dict(node_blob[0])
    orphan_entry["identifier"] = {"id": len(node_blob), "name": None}
    node_blob.append(orphan_entry)

    with pytest.warns(UserWarning, match="node"):
        reader.structure_with_context(structure, tensor_blob=tensor_blob, node_blob=node_blob)


def _attr_equal(original, reconstructed):
    if isinstance(original, torch.Tensor) or isinstance(reconstructed, torch.Tensor):
        return torch.equal(original, reconstructed)
    return original == reconstructed


@pytest.fixture
def scalar_value_node():
    from hippynn.graphs.nodes.base import ValueNode

    node = ValueNode(0.5)
    node.db_name = "custom_value_db_name"
    node.index_state = IdxType.Atoms
    return node

@pytest.fixture
def input_node():
    from hippynn.graphs.nodes.base import InputNode

    node = InputNode(db_name="my_input")
    node.index_state = IdxType.Atoms
    return node

@pytest.fixture
def loss_input_node():
    from hippynn.graphs.nodes.base import InputNode, LossInputNode

    origin = InputNode(db_name="origin")
    node = LossInputNode("loss_in", origin_node=origin)
    node.index_state = IdxType.Atoms
    return node

@pytest.fixture
def unary_node(scalar_value_node):
    from hippynn.graphs.nodes.base.base import NegNode
    node = NegNode(scalar_value_node)
    node.db_name = "custom_unary_db_name"
    return node

@pytest.fixture
def bin_node(scalar_value_node):
    node = scalar_value_node + scalar_value_node
    node.db_name = "custom_bin_db_name"
    return node

@pytest.fixture
def automodule_node():
    from hippynn.graphs.nodes.base import ValueNode
    from hippynn.graphs.nodes.indexers import FuzzyHistogrammer

    values = ValueNode(torch.linspace(0, 1, 10))
    node = FuzzyHistogrammer("hist", (values,), length=5, vmin=0.0, vmax=1.0)
    node.db_name = "custom_automodule_db_name"
    node.index_state = IdxType.Atoms
    return node


@pytest.fixture
def node(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "node",
    [
        pytest.param("scalar_value_node", id="ValueNode"),
        pytest.param("input_node", id="InputNode"),
        pytest.param("loss_input_node", id="LossInputNode"),
        pytest.param("unary_node", id="UnaryNode-NegNode"),
        pytest.param("bin_node", id="BinNode-AddNode"),
        pytest.param("automodule_node", id="AutoModule-FuzzyHistogrammer"),
    ],
    indirect=True,
)
def test_node_to_record_from_record_roundtrip(node):
    # Round-trips a single node through `to_record()`/`from_record()` in isolation
    # from rest of the serialization machinery
    record = node.to_record()
    rebuilt = type(node).from_record(name=node.name, parents=node.parents, **(record or {}))

    assert isinstance(rebuilt, type(node)), f"from_record returned {rebuilt!r} instead of a {type(node).__name__}"
    assert rebuilt.name == node.name
    assert rebuilt.parents == node.parents

    for attr in node.serialization_attributes or ():
        original, reconstructed = getattr(node, attr), getattr(rebuilt, attr)
        assert _attr_equal(original, reconstructed), f"{attr!r} mismatch: {original!r} != {reconstructed!r}"


def _roundtrip(obj):
    # Round-trips bare nodes/plain data through `CONTEXT_CONVERTER` directly.
    from hippynn.serializer.processor import CONTEXT_CONVERTER

    structure, tensor_blob, node_blob = CONTEXT_CONVERTER.unstructure_with_context(obj)
    rebuilt = CONTEXT_CONVERTER.structure_with_context(structure, tensor_blob=tensor_blob, node_blob=node_blob)
    return rebuilt, tensor_blob, node_blob


def test_serialize_simple_nodes():
    from hippynn.graphs.nodes.base import ValueNode
    from hippynn.graphs.nodes.base.base import AtLeast2D

    left = ValueNode(1.0)
    _rebuilt, _tensor_blob, node_blob = _roundtrip(left)
    assert len(node_blob) == 1

    right = ValueNode(2.0)
    combined = left + right

    tensor_node = ValueNode(torch.ones(3))
    _rebuilt, _tensor_blob, node_blob = _roundtrip(tensor_node)
    assert len(node_blob) == 1

    _rebuilt, _tensor_blob, node_blob = _roundtrip([combined, left, right])
    assert len(node_blob) == 3

    wrapped = AtLeast2D(combined)
    _rebuilt, _tensor_blob, node_blob = _roundtrip([left, right, combined, wrapped])
    assert len(node_blob) == 4


def test_serialize_automodule(processor_pair):
    from hippynn.graphs.nodes.base import ValueNode
    from hippynn.graphs.nodes.indexers import FuzzyHistogrammer

    writer, reader = processor_pair

    values = ValueNode(torch.linspace(0, 1, 10))
    node = FuzzyHistogrammer("hist", (values,), length=5, vmin=0.0, vmax=1.0)

    structure, tensor_blob, node_blob = writer.unstructure_with_context([values, node])
    restored_values, restored_node = reader.structure_with_context(structure, tensor_blob=tensor_blob, node_blob=node_blob)

    assert isinstance(restored_node, FuzzyHistogrammer)
    assert restored_node.parents[0] is restored_values
    assert restored_node.module_kwargs == {"length": 5, "vmin": 0.0, "vmax": 1.0}




def test_unstructure_record_shape(energy_graph):
    # Checks `unstructure_with_context` produces a `GraphModuleRecord`-shaped dict; not a round trip.
    from hippynn.serializer.processor import CONTEXT_CONVERTER

    record, tensor_data, _node_blob = CONTEXT_CONVERTER.unstructure_with_context(energy_graph)

    # `unstructure_with_context` always flattens a record-shaped result to a plain dict.
    assert isinstance(record["graph"], list)
    assert len(record["graph"]) > 0
    # `graph` entries are already-unstructured plain dicts (see `reference_node`), not live records.
    assert all(isinstance(info, dict) and "identifier" in info for info in record["graph"])
    assert record["_type"] == "graph_module_record"
    assert isinstance(tensor_data, list)



def test_protocol_mismatch_warns(tmp_path):
    from hippynn.graphs.nodes.base import ValueNode

    node = ValueNode(1.0)
    file = tmp_path / "node.pkl"
    hippynn.savegraph([node], str(file))

    box = pickle.loads(file.read_bytes())
    box["protocol_version"] -= 1
    file.write_bytes(pickle.dumps(box))

    with pytest.warns(UserWarning, match="protocol_version"):
        hippynn.loadgraph(str(file))


def test_tensor_count_mismatch_warns(tmp_path):
    from hippynn.graphs.nodes.base import ValueNode

    node = ValueNode(torch.ones(3))
    file = tmp_path / "node.pkl"
    weights_file = tmp_path / "node_weights.pt"
    hippynn.savegraph([node], str(file), weights_file=str(weights_file))

    tensor_data = torch.load(weights_file, weights_only=True)
    padded_tensor_data = list(tensor_data) + [torch.zeros(1)]
    torch.save(padded_tensor_data, weights_file)

    with pytest.warns(UserWarning, match="tensor"):
        hippynn.loadgraph(str(file), weights_file=str(weights_file))


def test_get_nodetype_rejects_namespace_root():
    # A node class's `__module__` must be a leaf module, not a namespace package root
    from hippynn.serializer.registry import get_nodetype

    class FakeGraphNode:
        pass

    FakeGraphNode.__module__ = "hippynn.graphs.nodes"
    with pytest.raises(ValueError, match="hippynn.graphs.nodes"):
        get_nodetype(FakeGraphNode)

    class FakeInterfaceNode:
        pass

    FakeInterfaceNode.__module__ = "hippynn.interfaces"
    with pytest.raises(ValueError, match="hippynn.interfaces"):
        get_nodetype(FakeInterfaceNode)


def test_serialize_valuenode_graph_module():
    from hippynn.graphs.nodes.base import ValueNode
    from hippynn.graphs import GraphModule
    import operator

    a_val = ValueNode(2)
    b_val = ValueNode(3)
    out_val = operator.add(a_val, b_val)

    graph = GraphModule([], [out_val])

    rebuilt, _tensor_blob, _node_blob = _roundtrip(graph)

    assert isinstance(rebuilt, GraphModule)
    assert rebuilt()[0] == graph()[0]


@ignore_sensitivity_warning
def test_serialize_energy_graph_module(input_nodes, energy_node, example_box):
    from hippynn.graphs import GraphModule

    # Use energy_node's own system_energy child, rather than the bare MultiNode, so
    # forward() returns a flat tuple of tensors.
    graph = GraphModule(input_nodes, [energy_node.system_energy])

    rebuilt, _tensor_blob, _node_blob = _roundtrip(graph)

    assert isinstance(rebuilt, GraphModule)

    original_state = graph.state_dict()
    rebuilt_state = rebuilt.state_dict()
    assert original_state.keys() == rebuilt_state.keys()
    for key in original_state:
        assert torch.equal(original_state[key], rebuilt_state[key])

    args = (example_box["species"], example_box["coordinates"], example_box["cell"])
    original_outputs = graph(*args)
    rebuilt_outputs = rebuilt(*args)
    for orig, new in zip(original_outputs, rebuilt_outputs):
        assert torch.allclose(orig, new)


@ignore_sensitivity_warning
@pytest.mark.parametrize("filetype", ["pkl", "pt", "json", "yaml"])
def test_savegraph_loadgraph_roundtrip(input_nodes, energy_node, example_box, tmp_path, filetype):
    from hippynn.graphs import GraphModule

    # Use energy_node's own system_energy child, rather than the bare MultiNode, so
    # forward() returns a flat tuple of tensors.
    graph = GraphModule(input_nodes, [energy_node.system_energy])

    main_file = tmp_path / f"graph.{filetype}"
    weights_file = tmp_path / "graph_weights.pt"

    hippynn.savegraph(graph, str(main_file), weights_file=str(weights_file))
    rebuilt = hippynn.loadgraph(str(main_file), weights_file=str(weights_file))

    assert isinstance(rebuilt, GraphModule)

    original_state = graph.state_dict()
    rebuilt_state = rebuilt.state_dict()
    assert original_state.keys() == rebuilt_state.keys()
    for key in original_state:
        assert torch.equal(original_state[key], rebuilt_state[key])

    args = (example_box["species"], example_box["coordinates"], example_box["cell"])
    original_outputs = graph(*args)
    rebuilt_outputs = rebuilt(*args)
    for orig, new in zip(original_outputs, rebuilt_outputs):
        assert torch.allclose(orig, new)


def test_savegraph_loadgraph_key_stability(tmp_path):
    # state_dict() keys must survive a save/load round trip even when node names collide.
    # Repeated 10x with fresh nodes since the name tie-break is hash-dependent (see _set_evaluation_order).
    from hippynn.graphs import GraphModule
    from hippynn.graphs.nodes.base import ValueNode
    from hippynn.graphs.nodes.indexers import FuzzyHistogrammer

    for attempt in range(10):
        values_a = ValueNode(torch.linspace(0.0, 1.0, 10))
        values_b = ValueNode(torch.linspace(0.0, 1.0, 10))

        hist_a = FuzzyHistogrammer("hist", (values_a,), length=4, vmin=0.0, vmax=1.0)
        hist_b = FuzzyHistogrammer("hist", (values_b,), length=4, vmin=5.0, vmax=10.0)

        graph = GraphModule([], [hist_a, hist_b])

        main_file = tmp_path / f"graph_{attempt}.pkl"
        weights_file = tmp_path / f"graph_weights_{attempt}.pt"
        hippynn.savegraph(graph, str(main_file), weights_file=str(weights_file))
        rebuilt = hippynn.loadgraph(str(main_file), weights_file=str(weights_file))

        original_state = graph.state_dict()
        rebuilt_state = rebuilt.state_dict()

        assert original_state.keys() == rebuilt_state.keys()
        for key in original_state:
            assert torch.equal(original_state[key], rebuilt_state[key]), (
                f"attempt {attempt}: key {key!r} disagrees between original and reloaded graph - "
                "moddict keys are assigned positionally, and the position is not stable when node names collide."
            )


def test_serialization_registry():
    from hippynn.serializer import ensure_registry_consistency

    # Test no problems
    ensure_registry_consistency(raise_=True)

    # Test problems when duplicate definitions clash
    from hippynn.graphs.nodes.base import InputNode
    class Foo(InputNode):
        pass
    class Foo(InputNode):
        pass
    with pytest.raises(ValueError):
        ensure_registry_consistency(raise_=True)


def test_late_registered_node_roundtrip(tmp_path):
    # A node class defined after `import hippynn` must be registered when a save/load session starts.
    from hippynn.graphs.nodes.base import ValueNode
    from hippynn.graphs.nodes.indexers import FuzzyHistogrammer
    from hippynn.serializer.registry import _NODE_REGISTRY

    class LateHistogrammer(FuzzyHistogrammer):
        pass

    assert LateHistogrammer not in _NODE_REGISTRY

    values = ValueNode(torch.linspace(0, 1, 10))
    node = LateHistogrammer("late_hist", (values,), length=5, vmin=0.0, vmax=1.0)
    file = tmp_path / "late.pkl"
    hippynn.savegraph([values, node], str(file))
    _restored_values, restored_node = hippynn.loadgraph(str(file))

    assert type(restored_node) is LateHistogrammer
    assert restored_node.module_kwargs == {"length": 5, "vmin": 0.0, "vmax": 1.0}



def test_shared_tensor_dedup():
    # Two nodes referencing the same tensor object must serialize it once.
    from hippynn.graphs.nodes.base import ValueNode

    shared = torch.ones(3)
    left = ValueNode(shared)
    right = ValueNode(shared)

    _rebuilt, tensor_blob, _node_blob = _roundtrip([left, right])

    assert len(tensor_blob) == 1, "the same tensor object referenced by two nodes must be stored once"


def test_multinode_attributes(energy_node):
    # check that child nodes of multinodes retain customized attributes
    from hippynn.graphs.indextypes import IdxType

    child = energy_node.mol_hier
    assert child.index_state == IdxType.Systems, "sanity check: this is the class-declared default before rewiring"
    assert child.db_name is None, "sanity check: no db_name is assigned by MultiNode.__init__"

    child.index_state = IdxType.Atoms
    child.db_name = "custom_mol_hier_name"

    rebuilt_child, _tensor_blob, _node_blob = _roundtrip(child)

    assert rebuilt_child.index_state == IdxType.Atoms, "rewired index_state must survive the round-trip"
    assert rebuilt_child.db_name == "custom_mol_hier_name", "rewired db_name must survive the round-trip"


def test_origin_node():
    from hippynn.graphs.nodes.base import InputNode, LossPredNode, LossTrueNode

    node = InputNode(db_name="X")
    pred_node, true_node = node.pred, node.true

    rebuilt_nodes, _tensor_blob, _node_blob = _roundtrip([pred_node, true_node])

    rebuilt_pred = next(n for n in rebuilt_nodes if isinstance(n, LossPredNode))
    rebuilt_true = next(n for n in rebuilt_nodes if isinstance(n, LossTrueNode))

    assert rebuilt_pred.origin_node is rebuilt_true.origin_node


@pytest.fixture
def written_energy_graph(energy_graph):
    from hippynn.serializer.processor import CONTEXT_CONVERTER

    return CONTEXT_CONVERTER.unstructure_with_context(energy_graph)


def _flatten_tensors(seq):
    for item in seq:
        if isinstance(item, (tuple, list)):
            yield from _flatten_tensors(item)
        else:
            yield item


@ignore_sensitivity_warning
def test_multioutput_roundtrip(written_energy_graph, energy_graph, example_box):
    # Round-trips a MultiNode-output graph via a pre-unstructured fixture, checking flattened forward outputs.
    from hippynn.serializer.processor import CONTEXT_CONVERTER

    record, tensor_data, node_blob = written_energy_graph
    rebuilt_graph = CONTEXT_CONVERTER.structure_with_context(record, tensor_blob=tensor_data, node_blob=node_blob)

    args = (example_box["species"], example_box["coordinates"], example_box["cell"])

    # energy_node is a bare MultiNode, so each entry of these outputs may itself be a
    # tuple of tensors (one per output_names) rather than a single tensor - flatten before comparing.
    original_outputs = list(_flatten_tensors(energy_graph(*args)))
    rebuilt_outputs = list(_flatten_tensors(rebuilt_graph(*args)))

    assert len(original_outputs) == len(rebuilt_outputs)
    for orig, new in zip(original_outputs, rebuilt_outputs):
        assert torch.allclose(orig, new)


@ignore_relocation
@ignore_weights_only_warning
@ignore_cusp_warning
@skip_if_no_models
def test_load_old():
    from hippynn.tools import active_directory
    from hippynn.experiment import load_checkpoint_from_cwd

    location = "./quad0_b512_int1_p5_GPU0_seed363144"
    with active_directory(MODEL_DIR / location, create=False):
        check = load_checkpoint_from_cwd(map_location="cpu")

    return


@ignore_weights_only_warning
@ignore_cusp_warning
@ignore_relocation
@ignore_sensitivity_warning
@skip_if_no_models
def test_run_old(example_box):
    from hippynn.tools import active_directory
    from hippynn.experiment import load_checkpoint_from_cwd
    from hippynn.graphs import Predictor

    location = "./quad0_b512_int1_p5_GPU0_seed363144"

    with active_directory(MODEL_DIR / location, create=False):
        check = load_checkpoint_from_cwd(map_location="cpu")

    model = check["training_modules"].model
    predictor = Predictor.from_graph(model)

    renamed_box = dict(
        Z=example_box["species"],
        R=example_box["coordinates"],
    )

    outputs = predictor(**renamed_box)

    return


@ignore_weights_only_warning
@ignore_cusp_warning
@ignore_relocation
@skip_if_no_models
def test_validate_old():

    from optimizer.test_configs import c2h6_config

    floatX = torch.float64
    # fmt: off
    expected_T = torch.tensor([[-566.04203682388606466702],
        [-633.75044483854321697436],
        [-683.59081489815389431897],
        [-706.40015799352158865076],
        [-713.39539090423795641982],
        [-711.64985486498540012690],
        [-705.40931757941268642753],
        [-694.82413721139664630755],
        [-686.65307032058126424090],
        [-677.45483365599477565411],
        [-571.35603221853000377450],
        [-471.07485521951696227916],
        [-485.98350140870610402999],
        [-526.74108660103354395687],
        [-525.06023249181453138590]], dtype=floatX)
    # fmt: on
    torch.set_printoptions(precision=24)
    
    from hippynn.tools import active_directory
    from hippynn.experiment import load_checkpoint_from_cwd
    from hippynn.graphs import Predictor

    location = "./quad0_b512_int1_p5_GPU0_seed363144"

    with active_directory(MODEL_DIR / location, create=False):
        check = load_checkpoint_from_cwd(map_location="cpu")

    model = check["training_modules"].model
    predictor = Predictor.from_graph(model).to(floatX)

    renamed_box = dict(
        Z=c2h6_config["Z"],
        R=c2h6_config["R"].to(floatX),
    )
    outputs = predictor(**renamed_box)
    out_energy = outputs["T"]

    absdiff = (out_energy - expected_T).abs()
    maxreldiff = (absdiff / expected_T).max()
    print(f"({maxreldiff=})")

    assert maxreldiff < 1.e-12, f"Output of network changed. (relative difference = {maxreldiff}, absdiff = {absdiff})"

    return

