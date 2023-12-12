import torch
from hippynn.graphs import GraphModule
from hippynn.graphs.nodes.inputs import SpeciesNode, PositionsNode, CellNode
from hippynn.graphs.nodes.indexers import acquire_encoding_padding
from hippynn.graphs.nodes.pairs import PeriodicPairIndexer


n_atom = 30
n_system = 30
n_dim = 3
distance_cutoff = 0.3

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

floatX = torch.float32

# Set up input nodes
sp = SpeciesNode("Z")
pos = PositionsNode("R")
cell = CellNode("C")

# Set up and compile calculation
enc, pidxer = acquire_encoding_padding(sp, species_set=[0, 1])
pairfinder = PeriodicPairIndexer("pair finder", (pos, enc, pidxer, cell), dist_hard_max=distance_cutoff)
computer = GraphModule([sp, pos, cell], [*pairfinder.children])
computer.to(device)

# Get some random inputs
species_tensor = torch.ones(n_system, n_atom, device=device, dtype=torch.int64)
pos_tensor = torch.rand(n_system, n_atom, 3, device=device, dtype=floatX)
cell_tensor = torch.eye(3, 3, device=device, dtype=floatX).unsqueeze(0).expand(n_system, n_dim, n_dim).clone()

# Run calculation
outputs = computer(species_tensor, pos_tensor, cell_tensor)

# Print outputs
output_as_dict = {c.name: o for c, o in zip(pairfinder.children, outputs)}
for k, v in output_as_dict.items():
    print(k, v.shape, v.dtype, v.min(), v.max())
