"""
Test module for verifying implementation correctness against pytorch.
"""
import numpy as np
import torch

from . import env_pytorch
from . import autograd_wrapper
from .utils import clear_pair_cache

import warnings


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


def get_simulated_data(n_molecules, n_atoms, atom_prob, n_features, n_nu, printinfo=False, dtype=None, device=torch.device("cpu")):
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

    # We'll just do fully connected molecules.
    pair_first_pre = np.repeat(atom_index[:, :, np.newaxis], n_atoms, axis=2)[np.nonzero(pair_presence)]
    pair_second_pre = np.repeat(atom_index[:, np.newaxis, :], n_atoms, axis=1)[np.nonzero(pair_presence)]

    pair_first = inv_real_atoms[pair_first_pre]
    pair_second = inv_real_atoms[pair_second_pre]
    n_pairs = len(pair_first)

    # NOTE: These fake sensitivities are NONSYMMETRIC.
    # Current HIP-NN does not do that, but a future one could.
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

    assert torch.allclose(pt_gradsense, ref_gradsense)
    assert torch.allclose(pt_gradfeat, ref_gradfeat)
    assert (pt_gradsense == ref_gradsense).all()
    assert (pt_gradfeat == ref_gradfeat).all()


test_pytorch_reference()
del test_pytorch_reference


# A class for testing the correctness of the kernels functions.


class Envops_tester:
    def __init__(self, envsum_raw, sensesum_raw, featsum_raw, suspicious_deviation=0.5):
        self.envsum, self.sensesum, self.featsum, *_ = autograd_wrapper.wrap_envops(
            envsum_impl=envsum_raw, sensesum_impl=sensesum_raw, featsum_impl=featsum_raw
        )
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

    def check_allclose_once(self, use_large=False, device=torch.device("cpu")):
        if use_large:
            sense, feat, pfirst, psecond = get_simulated_data(**TEST_LARGE_PARAMS, dtype=torch.float32, device=device)
        else:
            sense, feat, pfirst, psecond = get_simulated_data(**TEST_TINY_PARAMS, dtype=torch.float64, device=device)

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
            raise ValueError("Failed an operation on data with zero pairs") from ee
        print("Passed zero-pair check")

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
        self.check_empty(device=device)
        print("Checking gradients {} times...".format(n_grad))
        self.check_all_grad(repeats=n_grad, device=device)
        print("Passed gradient checks!")
        print("Checking forward methods on small data {} times...".format(n_small), flush=True)
        self.check_allclose(repeats=n_small, use_large=False, device=device)
        print("Passed small tensor forward checks!")
        print("Checking forward methods on large data {} times...".format(n_large), flush=True)
        self.check_allclose(repeats=n_large, use_large=True, device=device)
        print("Passed large tensor forward checks!")

    def check_speed(self, n_repetitions=10, device=torch.device("cpu"), data_size=TEST_LARGE_PARAMS, compare_against="pytorch"):

        if compare_against.lower() == "pytorch":
            comp_envsum = env_pytorch.envsum
            comp_sensesum = env_pytorch.sensesum
            comp_featsum = env_pytorch.featsum
        elif compare_against.lower() == "numba":
            comp_envsum = env_numba.new_envsum
            comp_sensesum = env_numba.new_sensesum
            comp_featsum = env_numba.new_featsum
        elif compare_against.lower() == "cupy":
            comp_envsum = env_cupy.cupy_envsum
            comp_sensesum = env_cupy.cupy_sensesum
            comp_featsum = env_cupy.cupy_featsum
        elif compare_against.lower() == "triton":
            comp_envsum = env_triton.envsum
            comp_sensesum = env_triton.featsum
            comp_featsum = env_triton.featsum

        else:
            raise ValueError("Unknown implementation to comapre against:'{}'".format(compare_against))

        te, ts, tf = (TimerHolder(name) for name in ("Envsum", "Sensesum", "Featsum"))
        tne, tns, tnf = (TimerHolder("{}_{}".format(compare_against, name)) for name in ("Envsum", "Sensesum", "Featsum"))

        print("Repetitions: {}".format(n_repetitions))
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
                torch.cuda.synchronize()
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
        print()  # Newline to terminate the ... printing
        for t in [tne, tns, tnf] + [te, ts, tf]:
            print("Mean {} time: {} Median: {}".format(t.name, t.mean_elapsed, t.median_elapsed))
        for tn, t in zip([tne, tns, tnf], [te, ts, tf]):
            print("{} Speedup: {}".format(t.name, tn.median_elapsed / t.median_elapsed))

        tnsum = sum(x.median_elapsed for x in [tne, tns, tnf])
        tsum = sum(x.median_elapsed for x in [te, ts, tf])
        print("Overall {} time: {}".format(compare_against, tnsum))
        print("Overall time now: {}".format(tsum))
        print("Overall speedup: {}".format(tnsum / tsum))
        return  # prof


import time


class TimerHolder:
    def __init__(self, name=None):
        self.snippets = []
        self.name = name

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


class TimedSnippet:
    def __init__(self):
        self.start = self.end = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end = time.time()

    @property
    def elapsed(self):
        return self.end - self.start


def main(env_impl, sense_impl, feat_impl, args=None):

    if args is None:
        # calling without arguments looks for them from command line
        args = parse_args()

    if isinstance(args, dict):
        from types import SimpleNamespace

        args = SimpleNamespace(**args)

    np.random.seed(args.seed)
    tester = Envops_tester(
        env_impl,
        sense_impl,
        feat_impl,
    )

    compare_against = args.compare_against
    test_gpu = not args.no_test_gpu
    test_cpu = not args.no_test_cpu
    correctness = not args.no_correctness

    if torch.cuda.is_available() and not args.no_test_gpu:
        print("Running GPU tests")
        free_mem, total_mem = torch.cuda.memory.mem_get_info()

        use_large_gpu = free_mem > 2**31
        use_verylarge_gpu = free_mem > 30 * (2**30)

        use_ultra = (not correctness) and use_verylarge_gpu and (compare_against.lower() != "pytorch")

        n_large_gpu = args.n_large if use_large_gpu else 0

        if correctness:
            tester.check_correctness(device=torch.device("cuda"), n_large=n_large_gpu)

        if use_verylarge_gpu:
            if use_ultra:

                print("-" * 80)
                print("Giga systems:", TEST_GIGA_PARAMS)
                tester.check_speed(
                    n_repetitions=20, data_size=TEST_GIGA_PARAMS, device=torch.device("cuda"), compare_against=compare_against
                )
                print("-" * 80)
                print("Ultra systems:", TEST_ULTRA_PARAMS)
                tester.check_speed(
                    n_repetitions=20, data_size=TEST_ULTRA_PARAMS, device=torch.device("cuda"), compare_against=compare_against
                )
            print("-" * 80)
            print("Mega systems:", TEST_MEGA_PARAMS)
            tester.check_speed(n_repetitions=20, data_size=TEST_MEGA_PARAMS, device=torch.device("cuda"), compare_against=compare_against)
        else:
            print("Numba indicates less than 30GB free GPU memory -- skipping mega system test")
        if use_large_gpu:
            print("-" * 80)
            print("Large systems:", TEST_LARGE_PARAMS)
            tester.check_speed(n_repetitions=20, data_size=TEST_LARGE_PARAMS, device=torch.device("cuda"), compare_against=compare_against)
        else:
            print("Numba indicates less than 2GB free GPU memory -- skipping large system test")

        print("-" * 80)
        print("Medium systems:", TEST_MEDIUM_PARAMS)
        tester.check_speed(n_repetitions=100, data_size=TEST_MEDIUM_PARAMS, device=torch.device("cuda"), compare_against=compare_against)
        print("-" * 80)
        print("Small systems:", TEST_SMALL_PARAMS)
        tester.check_speed(n_repetitions=100, data_size=TEST_SMALL_PARAMS, device=torch.device("cuda"), compare_against=compare_against)

    else:
        if not args.no_test_gpu:
            print("Cuda not available, not running GPU tests.")
        else:
            print("Skipped GPU tests.")

    if test_cpu:
        print("Running CPU tests")
        if correctness:
            tester.check_correctness(n_large=args.n_large)

        print("-" * 80)
        print("Large systems:", TEST_LARGE_PARAMS)
        tester.check_speed(n_repetitions=10, compare_against=compare_against)
        print("-" * 80)
        print("Medium systems:", TEST_MEDIUM_PARAMS)
        tester.check_speed(n_repetitions=100, data_size=TEST_MEDIUM_PARAMS, compare_against=compare_against)
        print("-" * 80)
        print("Small systems:", TEST_SMALL_PARAMS)
        tester.check_speed(n_repetitions=100, compare_against=compare_against, data_size=TEST_SMALL_PARAMS)
    else:
        print("Skipped CPU tests.")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="name for run")

    parser.add_argument(
        "--compare-against",
        type=str,
        default="pytorch",
        help="""
    implementation to compare speed with. Options are: pytorch, numba, cupy, triton""",
    )

    parser.add_argument(
        "--n_large",
        type=int,
        default=5,
        help="""
    Number of times to check correctness of forward pass. Set this to a large number (e.g. 200) to
    stress-test a new implementation against corner-cases.""",
    )

    parser.add_argument("--no-test-cpu", action="store_true", default=False, help="Set to false to skip CPU tests.")
    parser.add_argument("--no-test-gpu", action="store_true", default=False, help="Set to false to skip GPU tests.")
    parser.add_argument("--no-correctness", action="store_true", default=False, help="Set to false to skip GPU tests.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(
        env_numba.new_envsum,
        env_numba.new_sensesum,
        env_numba.new_featsum,
    )
