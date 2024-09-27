"""
Test module for verifying implementation correctness against pytorch.
"""
import platform
import warnings
import time

import numpy as np
import torch

from . import env_pytorch
from .utils import clear_pair_cache

from .registry import MessagePassingKernels

try:
    from . import env_numba
except ImportError:
    warnings.warn("numba implementation not importable.")
    env_numba = None
try:
    from . import env_cupy
except ImportError:
    warnings.warn("cupy implementation not importable.")
    env_cupy = None
try:
    from . import env_triton
except ImportError:
    warnings.warn("triton implementation not importable.")
    env_triton = None
try:
    from . import env_sparse
except ImportError:
    warnings.warn("sparse implementation not importable.")
    env_sparse = None


def get_simulated_data(
    n_molecules, n_atoms, atom_prob, n_features, n_nu, printinfo=False, dtype=None, device=torch.device("cpu"), randomize_order=False,
        use_duplicate=False,
):
    """
    Get semi-realistic test data for hipnn.
    n_molecules : number of molecules in the batch
    n_atoms     : number of atoms in each molecule
    atom_prob   : probability that an atom is real (not a padding)
    n_features  : number of features on e
    ach atom
    n_nu        : number of sensitivity functions

    Differences from real data:
        1)  Occupied atoms are (usually) first in the real arrays.
            This shouldn't matter because raw arrays are not output.

        2)  Sensitivity functions are randomly sparse with an
            average of 1 sensitivity non-zero per pair. Their amplitude is random.
            In real data, they are sparse, but 2 or 3 sequential ones will be on.
            Amplitudes have a concrete form based on distances
            Note: Sensitivities are not sparse here. There could be some use for
            representing them in a sparse way, though. I don't know how fast
            it would be to construct a sparse representation of sensitivities.

        3)  Sensitivity functions are not symmetric between pairs
            This necessitates summing over a pair (i,j) and (j, i) separately;
            certain recalculations are performed. I prefer code which works
            properly in the case of symmetric or non-symmetric sensitivities;
            we have some a case for asymmetric sensitivities (such as xACA),
            and in the future we may find use of nonsymmetric sensitivities.

        4)  Note that pairs never consist of the same atom repeatedly, that is,
            for all atoms i, (i,i) is not a pair.

    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    np_fdtype = torch.zeros(1, dtype=torch.get_default_dtype()).numpy().dtype
    molatom_shp = (n_molecules, n_atoms)
    molatom_presence = np.random.choice([False, True], p=[1 - atom_prob, atom_prob], size=molatom_shp)
    atom_presence = molatom_presence.reshape(n_atoms * n_molecules)

    mol_features = np.random.randn(n_molecules, n_atoms, n_features).astype(np_fdtype)
    mol_features[~molatom_presence] = 0

    atom_features = mol_features.reshape(n_molecules * n_atoms, n_features).astype(np_fdtype)[atom_presence]
    n_real = molatom_presence.sum()
    real_atoms_arange = np.arange(n_real, dtype=int)
    inv_real_atoms = np.zeros(n_molecules * n_atoms, dtype=int)
    inv_real_atoms[atom_presence] = real_atoms_arange
    # atom_index = np.arange(n_molecules*n_atoms).reshape(molatom_shp)
    # mol_index = np.repeat(np.arange(n_molecules)[:,np.newaxis],n_atoms,axis=1)

    # pair_dists = np.sqrt(((coords[:,:,np.newaxis] - coords[:,np.newaxis,:])**2).sum(axis=3))
    pair_presence = (molatom_presence[:, np.newaxis, :] * molatom_presence[:, :, np.newaxis]) & (
        ~np.identity(n_atoms, dtype=bool)[np.newaxis, :, :]
    )

    atom_index = np.arange(n_molecules * n_atoms).reshape(molatom_shp)
    nonzero_pair = np.nonzero(pair_presence)

    # We'll just do fully connected molecules.
    pair_first_pre = np.repeat(atom_index[:, :, np.newaxis], n_atoms, axis=2)[nonzero_pair]
    pair_second_pre = np.repeat(atom_index[:, np.newaxis, :], n_atoms, axis=1)[nonzero_pair]

    pair_first = inv_real_atoms[pair_first_pre]
    pair_second = inv_real_atoms[pair_second_pre]

    if randomize_order:
        random_order = np.random.permutation(np.arange(len(pair_first)))
        pair_first = pair_first[random_order]
        pair_second = pair_second[random_order]

    n_pairs = len(pair_first)

    # reduplicate for testing if an implementation works with duplicate pairs.
    if use_duplicate:
        pair_first[0] = pair_first[-1]
        pair_second[0] = pair_second[-1]

    # NOTE: These synthetic sensitivities are NONSYMMETRIC, that is, j->i does not mean i->j,
    # and also does not mean that the value of the sensitivity is the same
    # even when i<->j.

    on_sensitivites = np.random.choice([True, False], p=[3 / n_nu, 1 - 3 / n_nu], size=(n_pairs, n_nu))
    pair_sensitivites = np.random.random(size=(n_pairs, n_nu)) * on_sensitivites
    assert not (pair_first == pair_second).any()
    if printinfo:
        print("Number of molecules:", n_molecules)
        print("Number of atoms per molecule:", n_atoms)
        print("Fraction of real atoms:", atom_prob)
        print("Number of features:", n_features)
        print("Number of sensitivities:", n_nu)
        print("Total atoms:", n_real)
        print("Total pairs:", n_pairs)
        # print("Total (any) atom pairs:",n_atoms**2*n_molecules)
        # print("Total (nonblank) atom pairs:",(molatom_presence[:,np.newaxis,:]*molatom_presence[:,:,np.newaxis]).sum())
    pair_sense = torch.tensor(pair_sensitivites).to(dtype=dtype, device=device)
    features = torch.tensor(atom_features).to(dtype=dtype, device=device)
    pair_first = torch.tensor(pair_first).to(device=device)
    pair_second = torch.tensor(pair_second).to(device=device)
    return pair_sense, features, pair_first, pair_second


TEST_TINY_PARAMS = dict(n_molecules=2, n_atoms=3, atom_prob=1.0, n_features=5, n_nu=7)
TEST_SMALL_PARAMS = dict(n_molecules=10, n_atoms=30, atom_prob=0.7, n_features=10, n_nu=20)
TEST_MEDIUM_PARAMS = dict(n_molecules=100, n_atoms=30, atom_prob=0.7, n_features=20, n_nu=20)
TEST_LARGE_PARAMS = dict(n_molecules=1000, n_atoms=30, atom_prob=0.7, n_features=80, n_nu=20)
TEST_MEGA_PARAMS = dict(n_molecules=500, n_atoms=30, atom_prob=0.7, n_features=128, n_nu=100)
TEST_ULTRA_PARAMS = dict(n_molecules=500, n_atoms=30, atom_prob=0.7, n_features=128, n_nu=320)
TEST_GIGA_PARAMS = dict(n_molecules=32, n_atoms=30, atom_prob=0.7, n_features=512, n_nu=320)
TEST_PARAMS = dict(
    tiny=TEST_TINY_PARAMS,
    small=TEST_SMALL_PARAMS,
    medium=TEST_MEDIUM_PARAMS,
    large=TEST_LARGE_PARAMS,
    mega=TEST_MEGA_PARAMS,
    ultra=TEST_ULTRA_PARAMS,
    giga=TEST_GIGA_PARAMS,
)


# A class for testing the correctness and speed of the kernels function implementations.


class EnvOpsTester:
    def __init__(self, name: str, suspicious_deviation: int = 0.5):
        implementation = MessagePassingKernels.get_implementation(name)
        self.envsum = implementation.envsum
        self.sensesum = implementation.sensesum
        self.featsum = implementation.featsum
        self.name = name

        self.tol_f64 = dict(atol=1e-8, rtol=1e-5)
        self.tol_f32 = dict(atol=1e-5, rtol=1e-5)  # Absolute tolerance a bit fuzzier for float32.
        self.suspicious_deviation = suspicious_deviation

    def check_grad_and_gradgrad(self, func, differentiable_inputs, pair_first, pair_second, funcname=None):
        if funcname is None:
            funcname = func.__qualname__
        try:
            inputs = [x.requires_grad_(True) for x in differentiable_inputs]
            try:
                torch.autograd.gradcheck(func, [*inputs, pair_first, pair_second])
            except Exception as ee:
                raise ValueError(f"{funcname} failed grad check.") from ee
            try:
                torch.autograd.gradgradcheck(func, [*inputs, pair_first, pair_second])
            except Exception as ee:
                raise ValueError(f"{funcname} failed gradgrad check.") from ee
        finally:
            [x.requires_grad_(False) for x in inputs]

    def check_all_grad_once(self, device=torch.device("cpu")):
        sense, feat, pfirst, psecond = get_simulated_data(**TEST_TINY_PARAMS, dtype=torch.float64, device=device)
        n_atoms, n_features = feat.shape
        n_pairs, n_nu = sense.shape
        env = torch.rand(n_atoms, n_nu, n_features, dtype=sense.dtype, device=device)
        self.check_grad_and_gradgrad(self.envsum, (sense, feat), pfirst, psecond, funcname="envsum")
        self.check_grad_and_gradgrad(self.sensesum, (env, feat), pfirst, psecond, funcname="sensesum")
        self.check_grad_and_gradgrad(self.featsum, (env, sense), pfirst, psecond, funcname="featsum")

    def check_all_grad(self, repeats=3, device=torch.device("cpu")):
        for i in range(repeats):
            self.check_all_grad_once(device=device)

    def check_allclose_once(self, use_large=False, device=torch.device("cpu"), randomize_order=False):
        if use_large:
            params = TEST_LARGE_PARAMS
        else:
            params = TEST_TINY_PARAMS

        sense, feat, pfirst, psecond = get_simulated_data(
            **params,
            dtype=torch.float32,
            device=device,
            randomize_order=randomize_order,
        )

        n_atoms, n_features = feat.shape
        n_pairs, n_nu = sense.shape
        env = torch.rand(n_atoms, n_nu, n_features, dtype=sense.dtype, device=device)
        for name, fn, pythfn, in1, in2 in (
            ("envsum", self.envsum, env_pytorch.envsum, sense, feat),
            ("sensesum", self.sensesum, env_pytorch.sensesum, env, feat),
            ("featsum", self.featsum, env_pytorch.featsum, env, sense),
        ):
            args = in1, in2, pfirst, psecond

            try:
                max_deviation = self.all_close_witherror(fn(*args), pythfn(*args))
            except Exception as ee:
                raise RuntimeError("Failed during {}".format(name)) from ee
            if max_deviation > self.suspicious_deviation:
                print("Closeness check for {} by suspicious amount".format(name), max_deviation)

    def check_empty(self, device=torch.device("cpu")):

        sense, feat, pfirst, psecond = get_simulated_data(**TEST_TINY_PARAMS, dtype=torch.float64, device=device)
        pfirst = psecond = torch.zeros((0,), dtype=torch.long, device=pfirst.device)
        sense = torch.zeros((0, sense.shape[1]), dtype=sense.dtype, device=sense.device)

        try:
            env = self.envsum(sense, feat, pfirst, psecond)
            sense_g = self.sensesum(env, feat, pfirst, psecond)
            feat_g = self.featsum(env, sense, pfirst, psecond)
        except Exception as ee:
            raise ValueError("Failed an operation on data with zero pairs!") from ee
        print("Passed zero-pair check!")

    def check_forward_noerr(self, device=torch.device("cpu")):

        sense, feat, pfirst, psecond = get_simulated_data(**TEST_TINY_PARAMS, dtype=torch.float64, device=device)

        env = self.envsum(sense, feat, pfirst, psecond)
        sense_g = self.sensesum(env, feat, pfirst, psecond)
        feat_g = self.featsum(env, sense, pfirst, psecond)
        print("Passed forward execution check!")

    def all_close_witherror(self, r1, r2):
        r1 = r1.data.cpu().numpy()
        r2 = r2.data.cpu().numpy()
        if r1.dtype == np.float32:  # Get tolerance
            tol = self.tol_f32
        else:
            tol = self.tol_f64
        where = ~np.isclose(r1, r2, **tol)
        # np check formula: absolute(a - b) <= (atol + rtol * absolute(b))
        tol_arr = tol["atol"] + tol["rtol"] * np.abs(r2)
        diff_arr = np.abs(r1 - r2)
        violation = np.abs(diff_arr[where] / tol_arr[where])
        try:
            assert np.allclose(r1, r2, **tol)
        except AssertionError as aee:

            print("{} bad tolerances (of {})".format(np.sum(where), where.size))
            print("Allowed tolerance:")
            print(tol_arr[where])
            print("Observed deviations:")
            print(diff_arr[where])
            print("Ratio:")
            print(diff_arr[where] / tol_arr[where])
            print("Desired result:", r2[where])
            print("Observed result:", r1[where])
            print(r2.sum(), r1.sum())
            print(r1.sum(axis=1), r2.sum(axis=1))
            print(r1.sum(axis=0), r2.sum(axis=0))
            print("Locations:", np.nonzero(where))
            print("Violation stats: median: {}, max: {}".format(np.median(violation), np.max(violation)))
            raise aee
        return np.max(np.abs(diff_arr / tol_arr))  # max violation of bounds

    def check_allclose(self, repeats=30, use_large=False, device=torch.device("cpu")):
        from tqdm.auto import tqdm

        for i in tqdm(range(repeats), leave=False):
            try:
                self.check_allclose_once(use_large, device=device)
            except Exception as ee:
                raise RuntimeError("Failed during iteration {}".format(i)) from ee

    def check_correctness(self, n_grad=1, n_small=100, n_large=3, device=torch.device("cpu")):

        print("Checking that functions execute...")
        safe_synchronize()
        self.check_forward_noerr()

        safe_synchronize()
        self.check_empty(device=device)  # this is now covered in autograd wrapper...

        safe_synchronize()
        self.check_allclose_once(device=device, randomize_order=True)
        print("Passed random pair order test...")

        safe_synchronize()
        print("Checking gradients {} times...".format(n_grad))
        self.check_all_grad(repeats=n_grad, device=device)
        print("Passed gradient checks!")
        safe_synchronize()
        print("Checking forward methods on small data {} times...".format(n_small), flush=True)
        self.check_allclose(repeats=n_small, use_large=False, device=device)
        print("Passed small tensor forward checks!")
        safe_synchronize()
        print("Checking forward methods on large data {} times...".format(n_large), flush=True)
        self.check_allclose(repeats=n_large, use_large=True, device=device)
        print("Passed large tensor forward checks!")


    def check_speed(self, n_repetitions=10, device=torch.device("cpu"), data_size=TEST_LARGE_PARAMS, compare_against="pytorch"):

        comparison_impl = MessagePassingKernels.get_implementation(compare_against)
        comp_envsum = comparison_impl.envsum
        comp_sensesum = comparison_impl.sensesum
        comp_featsum = comparison_impl.featsum

        te, ts, tf = (TimerHolder(f"{self.name}_{name}", device=device) for name in ("Envsum", "Sensesum", "Featsum"))
        tne, tns, tnf = (TimerHolder(f"{compare_against}_{name}", device=device) for name in ("Envsum", "Sensesum", "Featsum"))

        print(f"Repetitions: {n_repetitions}")
        with torch.autograd.no_grad():
            # Warming up by running on data of this specific size
            sense, feat, pfirst, psecond = get_simulated_data(**data_size, dtype=torch.float32, device=device)
            env = comp_envsum(sense, feat, pfirst, psecond)
            comp_sensesum(env, feat, pfirst, psecond)
            comp_featsum(env, sense, pfirst, psecond)
            self.envsum(sense, feat, pfirst, psecond)
            self.sensesum(env, feat, pfirst, psecond)
            self.featsum(env, sense, pfirst, psecond)

            # Note: in this implementation we clear the pair cache for each run.
            # In real conditions speedups could be greater due to caching of pairs.

            # with torch.autograd.profiler.profile() as prof:
            for i in range(n_repetitions):
                print(".", end="", flush=True)
                sense, feat, pfirst, psecond = get_simulated_data(**data_size, dtype=torch.float32, device=device)
                with tne.add():
                    env = comp_envsum(sense, feat, pfirst, psecond)
                with tns.add():
                    comp_sensesum(env, feat, pfirst, psecond)
                with tnf.add():
                    comp_featsum(env, sense, pfirst, psecond)
                clear_pair_cache()
                with te.add():
                    self.envsum(sense, feat, pfirst, psecond)
                with ts.add():
                    self.sensesum(env, feat, pfirst, psecond)
                with tf.add():
                    self.featsum(env, sense, pfirst, psecond)
        print()  # Newline to terminate the '...' printing
        for t in [tne, tns, tnf] + [te, ts, tf]:
            print("Mean {} time: {} Median: {}".format(t.name, t.mean_elapsed, t.median_elapsed))
        for tn, t in zip([tne, tns, tnf], [te, ts, tf]):
            print("{} Speedup: {}".format(t.name, tn.median_elapsed / t.median_elapsed))

        tnsum = sum(x.median_elapsed for x in [tne, tns, tnf])
        tsum = sum(x.median_elapsed for x in [te, ts, tf])
        print("Overall {} time: {}".format(compare_against, tnsum))
        print("Overall time now: {}".format(tsum))
        print("Overall speedup: {}".format(tnsum / tsum))
        new_results = [t.to_dict() for t in [tne, tns, tnf]]
        compare_results = [t.to_dict() for t in [te, ts, tf]]
        return new_results, compare_results


def safe_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return


class TimerHolder:
    def __init__(self, name, device):
        self.snippets = []
        self.name = name
        if device.type == "cuda":
            self.device = torch.cuda.get_device_name(device)
        elif device.type == "cpu":
            self.device = platform.processor()
        else:
            self.device = device.type

    def add(self):
        t = TimedSnippet()
        self.snippets.append(t)
        return t

    @property
    def elapsed(self):
        return sum(t.elapsed for t in self.snippets)

    @property
    def mean_elapsed(self):
        return self.elapsed / len(self.snippets)

    @property
    def median_elapsed(self):
        return np.median([t.elapsed for t in self.snippets])

    def to_dict(self):
        return {
            "name": self.name,
            "count": len(self.snippets),
            "mean": self.mean_elapsed,
            "median": self.median_elapsed,
            "individual": [t.elapsed for t in self.snippets],
            "device": self.device,
        }


class TimedSnippet:
    def __init__(self):
        self.start = self.end = None

    def __enter__(self):
        safe_synchronize()
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        safe_synchronize()
        self.end = time.time()

    @property
    def elapsed(self):
        return self.end - self.start


def speed_test_loop(tester: EnvOpsTester, speed_test_spec: dict, device: torch.device, compare_against: str):
    all_results = {}
    for sys_type, repetitions in speed_test_spec.items():
        params = TEST_PARAMS[sys_type]
        print("-" * 80)
        print(f"{sys_type} systems:", params)
        results = tester.check_speed(n_repetitions=repetitions, data_size=params, device=device, compare_against=compare_against)
        all_results[sys_type] = results
    return results


def main(args=None):

    if args is None:
        # calling without arguments looks for them from command line
        args = parse_args()

    if isinstance(args, dict):
        from types import SimpleNamespace

        args = SimpleNamespace(**args)

    np.random.seed(args.seed)

    tester = EnvOpsTester(name=args.implementation)

    test_gpu = not args.no_gpu
    device = torch.device(args.accelerator)
    if test_gpu:
        device_type = device.type
        if device_type == "cuda":
            available = torch.cuda.is_available()
        elif device_type == "mps":
            available = torch.backends.mps.is_available()
        elif device_type == "cpu":
            raise ValueError("Accelerator cannot be set to 'cpu'.")
        else:
            available = True  # hopefully...

        if not available:
            print(f"Backend {device_type} not available, skipping GPU tests.")
            test_gpu = False

    test_cpu = not args.no_cpu
    speed = not args.no_speed
    compare_against = args.compare_against
    if speed:
        # just so that we error early if a bad name is specified.
        compare_impl = MessagePassingKernels.get_implementation(args.compare_against)
    correctness = not args.no_correctness

    # Standard test numbers.
    speed_tests = dict(mega=3, large=5, medium=100, small=100)
    # if compare_against == 'pytorch':
    #     del speed_tests['mega']  # This takes about 15s/iteration on CPU

    if test_gpu:
        print("Running GPU tests.")
        gpu_speed_tests = speed_tests.copy()
        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.memory.mem_get_info()
            del total_mem
        else:
            free_mem = 0  # Don't assume there is a lot of memory.

        # More than 2GB memory, we can run large systems
        use_large_gpu = free_mem > 2**31
        if not use_large_gpu:
            print("Torch indicates less than 2GB free GPU memory -- skipping large system test")
            del gpu_speed_tests["large"]
        else:
            gpu_speed_tests["large"] = 20

        # More than 30GB memory, we can run mega and ultra tests
        use_verylarge_gpu = free_mem > 30 * (2**30)
        if use_verylarge_gpu:
            gpu_speed_tests.pop("mega")
            gpu_speed_tests = dict(mega=20, **gpu_speed_tests)
        else:
            gpu_speed_tests.pop("mega", None)
            print("Numba indicates less than 30GB free GPU memory -- skipping mega system test")

        # Note: base pytorch implementation will error on ultra configurations.
        use_ultra = (not correctness) and use_verylarge_gpu and (compare_against.lower() != "pytorch")
        if use_ultra:
            gpu_speed_tests = dict(giga=20, ultra=20, **gpu_speed_tests)

        if correctness:
            n_large_gpu = args.n_large if use_large_gpu else 0
            tester.check_correctness(device=device, n_large=n_large_gpu)
        else:
            print("Skipped correctness checks.")

        if speed:
            speed_test_loop(tester, gpu_speed_tests, device=device, compare_against=compare_against)
        else:
            print("Skipped speed tests.")
    else:
        print("Skipped GPU tests.")

    if test_cpu:
        print("Running CPU tests.")
        if correctness:
            tester.check_correctness(n_large=args.n_large)
        else:
            print("Skipped correctness checks.")

        if speed:
            speed_test_loop(tester, speed_tests, device=torch.device("cpu"), compare_against=compare_against)
        else:
            print("Skipped speed tests.")
    else:
        print("Skipped CPU tests.")


# reference implementation
def test_pytorch_reference():
    sense, feat, pfirst, psecond = get_simulated_data(**TEST_TINY_PARAMS, dtype=torch.float64)
    sense.requires_grad_(True)
    feat.requires_grad_(True)
    env = env_pytorch.envsum(sense, feat, pfirst, psecond)
    pt_gradsense, pt_gradfeat = torch.autograd.grad(env.sum(), [sense, feat])
    sense.requires_grad_(False)
    feat.requires_grad_(False)
    ref_gradsense = env_pytorch.sensesum(torch.ones_like(env), feat, pfirst, psecond)
    ref_gradfeat = env_pytorch.featsum(torch.ones_like(env), sense, pfirst, psecond)

    # Note: Numerical differences
    assert torch.allclose(pt_gradsense, ref_gradsense, atol=1e-15, rtol=1e-15)
    assert torch.allclose(pt_gradfeat, ref_gradfeat, atol=1e-15, rtol=1e-15)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("implementation", type=str, help="Implementation to test.")

    parser.add_argument(
        "--compare-against",
        type=str,
        default="pytorch",
        help="""
    Implementation to compare speed with. Options are: pytorch, numba, cupy, triton
    Correctness is always compared against pytorch.
    """,
    )

    parser.add_argument("--accelerator", type=str, default="cuda", help="Device to treat as the GPU.")
    parser.add_argument("--no-cpu", action="store_true", default=False, help="Flag to skip CPU tests.")
    parser.add_argument("--no-gpu", action="store_true", default=False, help="Flag to skip GPU tests.")
    parser.add_argument("--no-speed", action="store_true", default=False, help="Flag to skip speed tests.")
    parser.add_argument("--no-correctness", action="store_true", default=False, help="Flag to skip correctness tests.")

    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument(
        "--n_large",
        type=int,
        default=5,
        help="""
    Number of times to check correctness of forward pass. Set this to a large number (e.g. 200) to
    stress-test a new implementation against corner-cases.
    """,
    )


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Ensure nothing is going wrong with reference sensesum and featsum
    test_pytorch_reference()

    main()
