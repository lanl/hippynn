import dataclasses

import pytest

import hippynn

from conftest import skip_if_no_models, MODEL_DIR
from conftest import ignore_cusp_warning, ignore_relocation, ignore_weights_only_warning, ignore_sensitivity_warning
import torch


@pytest.fixture
def processor_pair():
    """A fresh writer/reader Processor pair, mirroring separate serialize/deserialize sessions."""
    from hippynn.serializer.processor import ContextConverter

    return ContextConverter(), ContextConverter()


def test_processor_plain_data_roundtrip(processor_pair):
    writer, reader = processor_pair

    data = {"a": 1, "b": [1, 2.5, "three", None, (4, 5)], "c": {"nested": True}}

    structure, tensor_blob = writer.unstructure(data)
    assert tensor_blob == {}

    restored = reader.structure(structure, object, tensor_blob=tensor_blob)
    assert restored == data


def test_processor_single_tensor_roundtrip(processor_pair):
    writer, reader = processor_pair

    data = {"weight": torch.arange(6).reshape(2, 3)}

    structure, tensor_blob = writer.unstructure(data)
    assert len(tensor_blob) == 1

    restored = reader.structure(structure, object, tensor_blob=tensor_blob)
    assert torch.equal(restored["weight"], data["weight"])


def test_processor_shared_tensor_identity_roundtrip(processor_pair):
    writer, reader = processor_pair

    shared = torch.ones(3)
    data = {"a": shared, "b": shared}

    structure, tensor_blob = writer.unstructure(data)
    assert len(tensor_blob) == 1, "the same tensor object referenced twice must be stored once"

    restored = reader.structure(structure, object, tensor_blob=tensor_blob)
    assert restored["a"] is restored["b"], "shared tensor identity must survive the round-trip"


def test_processor_set_roundtrip(processor_pair):
    writer, reader = processor_pair

    data = {"values": {1, 2, 3}}

    structure, tensor_blob = writer.unstructure(data)
    restored = reader.structure(structure, object, tensor_blob=tensor_blob)

    assert restored["values"] == data["values"]


def test_processor_embedded_object_reference_roundtrip(processor_pair):

    from hippynn.serializer.processor import ContextConverter

    writer, reader = processor_pair

    @dataclasses.dataclass
    class Leaf:
        payload: str

    @dataclasses.dataclass
    class Root:
        leaves: list[Leaf]

    @ContextConverter.register_unstructure(Leaf)
    def _unstructure_leaf(obj, *, ctx):
        blob = getattr(ctx, "_leaf_blob", None)
        if blob is None:
            blob, ids = {}, {}
            ctx._leaf_blob, ctx._leaf_ids = blob, ids
        key = ctx._leaf_ids.get(id(obj))
        if key is None:
            key = len(blob)
            ctx._leaf_ids[id(obj)] = key
            blob[key] = obj.payload
        return {"kind": "leaf_ref", "key": key}

    @ContextConverter.register_structure(Leaf)
    def _structure_leaf(data, typ, *, ctx):
        cache = getattr(ctx, "_leaf_objects", None)
        if cache is None:
            cache = {}
            ctx._leaf_objects = cache
        key = data["key"]
        if key not in cache:
            cache[key] = typ(ctx._leaf_blob[key])
        return cache[key]

    shared_leaf = Leaf(payload="shared")
    root = Root(leaves=[shared_leaf, shared_leaf])

    structure, tensor_blob = writer.unstructure(root)
    reader._leaf_blob = writer._leaf_blob  # out-of-band leaf payloads, mirroring tensor_blob's write->read hand-off
    restored = reader.structure(structure, Root, tensor_blob=tensor_blob)

    assert restored.leaves[0] is restored.leaves[1], "repeated references to one Leaf must resolve to one object"
    assert restored.leaves[0].payload == "shared"


def test_serialize_autonode_graph(energy_graph):
    from hippynn.serializer import serialize, NodeInfo, SerializedGraphModule

    record, tensor_data = serialize(energy_graph)

    assert isinstance(record.graph.node_list, list)
    assert len(record.graph.node_list) > 0
    assert all(isinstance(info, NodeInfo) for info in record.graph.node_list)
    assert isinstance(record.module, SerializedGraphModule)
    assert isinstance(tensor_data, dict)
    

def test_serialize_simple_nodes():
    from hippynn.serializer import serialize_node_list
    from hippynn.graphs.nodes.base import ValueNode

    value = ValueNode(1.0)

    serial_data, tensor_data, identifier_map = serialize_node_list([value])
    assert len(serial_data.node_list) == 1

    left = ValueNode(1.0)
    right = ValueNode(2.0)
    combined = left + right

    serial_data, tensor_data, identifier_map = serialize_node_list([left, right, combined])
    assert len(serial_data.node_list) == 3


def test_serialize_automodule_kwargs_roundtrip(processor_pair):
    """AutoModule.from_record's tuple-form auto_module_kwargs translation, exercised directly."""
    from hippynn.graphs.nodes.base import ValueNode
    from hippynn.graphs.nodes.indexers import FuzzyHistogrammer

    writer, reader = processor_pair

    values = ValueNode(torch.linspace(0, 1, 10))
    node = FuzzyHistogrammer("hist", (values,), length=5, vmin=0.0, vmax=1.0)

    structure, tensor_blob = writer.unstructure([values, node])
    reader.node_blob = writer.node_blob  # out-of-band node records, mirroring tensor_blob's write->read hand-off
    restored_values, restored_node = reader.structure(structure, object, tensor_blob=tensor_blob)

    assert isinstance(restored_node, FuzzyHistogrammer)
    assert restored_node.parents[0] is restored_values
    assert restored_node.module_kwargs == {"length": 5, "vmin": 0.0, "vmax": 1.0}


def test_serialize_atleast2d_and_arithmetic_leaf_coverage():
    from hippynn.serializer import serialize_node_list
    from hippynn.graphs.nodes.base import ValueNode
    from hippynn.graphs.nodes.base.base import AtLeast2D

    left = ValueNode(1.0)
    right = ValueNode(2.0)
    combined = left + right
    wrapped = AtLeast2D((combined,))

    serial_data, tensor_data, identifier_map = serialize_node_list([left, right, combined, wrapped])
    assert len(serial_data.node_list) == 4


def test_serialize_shared_tensor_across_nodes_dedup():
    from hippynn.serializer import serialize_node_list
    from hippynn.graphs.nodes.base import ValueNode

    shared = torch.ones(3)
    left = ValueNode(shared)
    right = ValueNode(shared)

    serial_data, tensor_data, identifier_map = serialize_node_list([left, right])

    assert len(tensor_data) == 1, "the same tensor object referenced by two nodes must be stored once"


def test_serialize_origin_node_roundtrip(energy_node, energy_graph, example_box):
    from hippynn.graphs.nodes.loss import MAELoss
    from hippynn.serializer import serialize, deserialize
    from hippynn.graphs.nodes.base import get_connected_nodes, LossPredNode, LossTrueNode

    loss_node = MAELoss.of_node(energy_node)

    record, tensor_data = serialize(loss_node)
    rebuilt_loss = deserialize(record, tensor_data)

    rebuilt_nodes = get_connected_nodes([rebuilt_loss], descendants=False, ancestors=True)
    rebuilt_pred = next(n for n in rebuilt_nodes if isinstance(n, LossPredNode))
    rebuilt_true = next(n for n in rebuilt_nodes if isinstance(n, LossTrueNode))

    assert rebuilt_pred.origin_node is rebuilt_true.origin_node


@pytest.fixture
def written_energy_graph(energy_graph):
    from hippynn.serializer import serialize

    return serialize(energy_graph)


def test_serialize_roundtrip(written_energy_graph, energy_graph, example_box):
    record, tensor_data = written_energy_graph

    from hippynn.serializer import deserialize

    rebuilt_graph = deserialize(record, tensor_data)

    args = (example_box["species"], example_box["coordinates"], example_box["cell"])

    original_outputs = energy_graph(*args)
    rebuilt_outputs = rebuilt_graph(*args)

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

