import torch, tqdm
import concurrent.futures
import numpy as np
from pathlib import Path
from ._settings_setup import settings

class ArrDict():
    def __init__(self, d):
        self.d = d
    
    def update(self, nd):
        ###!!! This is NOT a symmetrical operation!
        assert set(self.d.keys()) == set(nd.d.keys())
        ret = {k: (nd.d[k] if self.d[k] is None else np.concatenate((self.d[k], nd.d[k])))\
                for k in self.d.keys()}
        return ArrDict(ret)
    
def N_to_m_batches(N, m):
    q = N // m
    r = N % m
    batch_indices = []
    start = 0
    l = list(range(N))
    for i in range(m):
        end = start + q + (1 if i<r else 0)
        batch_indices.append((i, l[start:end]))
        start = end
    
    return batch_indices

def load_hipnn_folder(model_dir, device='cpu', verbose=False, return_training_modules=False):
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
    structure = torch.load(model_dir/'experiment_structure.pt', map_location=device)
    state = torch.load(model_dir/'best_model.pt', map_location=device)
    structure["training_modules"].model.load_state_dict(state)

    if verbose:
        print(structure["training_modules"])

    return structure["training_modules"] if return_training_modules else structure["training_modules"].model

def rename_nodes(model):
    from .graphs import inputs, targets
    from .graphs import find_unique_relative

    species_node = find_unique_relative(model.nodes_to_compute, inputs.SpeciesNode)
    species_node.db_name="Z"
    pos_node = find_unique_relative(model.nodes_to_compute, inputs.PositionsNode)
    pos_node.db_name="R"

    energy_node = None
    for node in model.nodes_to_compute:
        if node.name.endswith('.mol_energy'):
            energy_node = node
        elif node.name == 'gradients':
            # Although it is called gradient node, the output is already forces
            force_node = node
            force_node.set_dbname("F")   
    if energy_node is None:
        energy_node = find_unique_relative(model.nodes_to_compute, targets.HEnergyNode)
    
    # energy_node.db_name='T' does not work here
    energy_node.set_dbname("T")

    return model


def multiGPU(f):
    # decorator to enable multi-GPU parallelization of an inference function
    # assume the input function takes Z, R, batch_size, device and other arguments as input
    # in which Z, R are padded numbers/coords tensors of N conformers, N is large but you have m GPUs
    # so you want to separate Z/R to m parts and run on different GPUs
    # assume the input function f would return a dictionary of numpy arraies
    def g(Z, R, batch_size=1024, device=-1, **kwargs):
        if device != -1:
            # if device != -1, the decorated function should behave just like 
            # how it would work without this decorator
            return f(Z=Z, R=R, batch_size=batch_size, device=device, **kwargs)
        else:
            # if device == -1, map inference tasks evenly to all GPUs it can find
            # set CUDA_VISIBLE_DEVICE to ignore some GPU 
            N_GPU = torch.cuda.device_count()
            assert Z.shape[0] == R.shape[0]
            N_mol = Z.shape[0]

            assignments = []
            for gpu_id, indices in N_to_m_batches(N_mol, N_GPU):
                device = 'cuda:%d'%(gpu_id)
                z = Z[indices].clone().detach()
                r = R[indices].clone().detach()
                assignment = {"Z":z, "R":r, "batch_size":batch_size, "device":device}
                assignment.update(kwargs)
                assignments.append( (gpu_id, assignment) )

            with concurrent.futures.ThreadPoolExecutor() as executor:
                tasks = [executor.submit(lambda x: (x[0], f( **x[1] )), inp) for inp in assignments]
                # Note that outputs may need sorting
                outputs = dict([t.result() for t in tasks])

        ret = None
        for i in range(N_GPU):
            ret = ArrDict(outputs[i]) if ret is None else ret.update( ArrDict(outputs[i]) )
        
        return ret.d
    
    return g

@multiGPU
def batch_inference(hipnn_predictor, Z, R, model_loader=None, predictor_loader=None, Z_name='Z', R_name='R', to_collect='T', batch_size=1024, device='cpu', no_grad=True):
    # assume Z, R are torch.tensor in the dtype that hipnn_predictor can accept
    # Z is in the shape (N_samples, molcule_size), R is in the shape (N_samples, molcule_size, 3)
    # they should be padded already and can locate on cpu, output would be dict of np.array
    # note that sometimes Z/R have different input names so you need to specify them
    if isinstance(hipnn_predictor, Path) or isinstance(hipnn_predictor, str):
        if model_loader is None:
            model = load_hipnn_folder(model_dir=hipnn_predictor, device=device)
            model = rename_nodes(model)
        if no_grad:
            model.requires_grad_=False
        if predictor_loader is None:
            from .graphs import Predictor
            hipnn_predictor = Predictor.from_graph(model, model_device=device, return_device=device)
        else: 
            hipnn_predictor = predictor_loader(model)
    else:
        ###TODO: I'm not sure if there is a way to move loaded hipnn predictor to a different device
        # if so we can take a predictor on cpu as input, copy it and move to different devices
        # instead of loading it from scratch to each device when multiGPU is enabled
        raise NotImplementedError

    if to_collect is None:
        ret = {}
        for k in hipnn_predictor.out_names:
            if k.endswith('.mol_energy'):
                ret[k] = []
        print('No targets specified, auto-detected the following energy-related keywords:' )
        print(list(ret.keys()))
    else:
        ret = {k:[] for k in to_collect}

    N = Z.shape[0]
    for start_idx in (range(0, N, batch_size) if settings.PROGRESS is None else tqdm.tqdm(range(0, N, batch_size))):
        end_idx = min(start_idx+batch_size, N)

        z = Z[start_idx:end_idx].to(device)
        r = R[start_idx:end_idx].to(device)
        batch_ret = hipnn_predictor(**{Z_name:z, R_name:r})

        for k in ret.keys():
            ret[k].append(batch_ret[k].detach().cpu().numpy())
            
    for k,v in ret.items():
        ret[k] = np.concatenate(v)

    torch.cuda.empty_cache()

    return ret

@multiGPU
def batch_optimize(loaded_optimizer, max_steps, Z, R, opt_algorithm='FIRE', 
                   batch_size=512, device='cpu', force_key='F', force_sign=1.0, return_coords=False):
    
    if isinstance(loaded_optimizer, Path) or isinstance(loaded_optimizer, str):
        from .optimizer import batch_optimizer, algorithms
        model = load_hipnn_folder(loaded_optimizer, device=device)
        model = rename_nodes(model)
        loaded_optimizer = batch_optimizer.Optimizer(model, \
            algorithm=getattr(algorithms, opt_algorithm)(max_steps=max_steps, device=device), 
            force_key=force_key, force_sign=force_sign, 
            dump_traj=False, device=device, relocate_optimizer=True)
    else:
        ###TODO: same question as in batch_inference
        raise NotImplementedError
        
    N = Z.shape[0]

    optimized_energies = []
    optimized_coords = []

    for start_idx in (range(0, N, batch_size) if settings.PROGRESS is None else tqdm.tqdm(range(0, N, batch_size))):
        end_idx = min(start_idx+batch_size, N)

        z = Z[start_idx:end_idx].to(device)
        r = R[start_idx:end_idx].to(device)
        opt_coord, model_ret = loaded_optimizer(Z=z, R=r)
        optimized_energies.append(model_ret['T'].detach().cpu().numpy())
        if return_coords:
            optimized_coords.append(opt_coord.detach().cpu().numpy())

        torch.cuda.empty_cache()

    optimized_energies = np.concatenate(optimized_energies)

    if return_coords:
        optimized_coords = np.concatenate(optimized_coords)
        return {"optimized_energies":optimized_energies.reshape(-1,), "optimized_coords":optimized_coords}
    
    return {"optimized_energies":optimized_energies.reshape(-1,)}