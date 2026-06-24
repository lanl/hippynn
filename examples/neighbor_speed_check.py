
import torch
from hippynn.layers.pairs.csr_pairs.neighbor_algorithm import calc_neighbors
import time
from tqdm.auto import tqdm



run_original_for_comparion=True
use_cuda = torch.cuda.is_available()
# use_cuda = False
n_warmup = 3
n_trials = 5
n_atoms = 1000
n_systems = 10
expected_neighbors_per_atom = 100
# Build cutoff so that neighbor count is according to the above; V = 4/3 pi r^#
cutoff = (1*expected_neighbors_per_atom*3/(4*n_atoms*torch.pi))**(1/3)
print(f"{n_systems=}")
print(f"{n_atoms=}")
print(f"{cutoff=}")


# Note: These functions close over script variables!!
def bench_once(function):
    positions = torch.rand(n_systems,n_atoms,3,device=device)
    nonblank = torch.ones((n_systems,n_atoms),dtype=torch.long,device=device)
    cells = torch.eye(3,device=device).unsqueeze(0).expand(n_systems,3,3)

    sync()
    start = time.time()

    pair_info = function(positions,nonblank,cells,cutoff)
    sync()
    n_pairs = pair_info[0].shape[0]
    n_pairs_per_atom = n_pairs/(n_atoms*n_systems)

    return time.time() - start, n_pairs_per_atom


def benchmark(function):
    print("benchmarking", function)
    for _ in tqdm(range(n_warmup),unit="warmup runs", leave=True):
        bench_once(function)
    times = []
    for i in tqdm(range(n_trials),unit="benchmark runs", leave=True):
        t,n_pairs_per_atom = bench_once(function)
        times.append(t)

    total_time = 1000*sum(times)/len(times) # in milliseconds
    time_per_atom = 1000*total_time/(n_atoms*n_systems) # in microseconds
    print(f"{n_pairs_per_atom=}")
    print(f"{total_time=} ms")
    print(f"{time_per_atom=} μs")


if use_cuda:
    device = "cuda"
    def sync(): torch.cuda.synchronize()
else:
    device = "cpu"
    def sync(): pass

benchmark(calc_neighbors)

if run_original_for_comparion:
    
    from hippynn.graphs import GraphModule
    from hippynn.graphs.nodes.inputs import SpeciesNode, PositionsNode, CellNode
    from hippynn.graphs.nodes.indexers import acquire_encoding_padding
    from hippynn.graphs.nodes.pairs import PeriodicPairIndexer

    floatX = torch.float32

    # Set up input nodes
    sp = SpeciesNode("Z")
    pos = PositionsNode("R")
    cell = CellNode("C")
    # Set up and compile calculation
    enc, pidxer = acquire_encoding_padding(sp, species_set=[0, 1])
    pairfinder = PeriodicPairIndexer("pair finder", (pos, enc, pidxer, cell), dist_hard_max=cutoff)
    computer = GraphModule([sp, pos, cell], [*pairfinder.children])
    computer.to(device)

    def original_implementation(positions,nonblank,cells,cutoff):
        species_tensor = nonblank.to(torch.long)
        return computer(species_tensor, positions, cells)

    benchmark(original_implementation)
