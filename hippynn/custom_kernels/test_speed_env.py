import json
import pathlib

import numpy as np
import torch
from .registry import MessagePassingKernels

from .test_env import TEST_PARAMS, EnvOpsTester


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("implementations", type=str, nargs="*", help="Implementation(s) to test.")
    parser.add_argument("--seed", type=int, default=0, help="Seed")

    parser.add_argument("--all-hidden", action="store_true", default=False, help="Use all implementations, even with _ beginning.")
    parser.add_argument("--all-impl", action="store_true", default=False, help="Use all non-hidden implementations.")
    parser.add_argument("--all-gpu", action="store_true", default=False, help="Use low-mem implementations suitable for GPU.")
    parser.add_argument("--all-cpu", action="store_true", default=False, help="CPU-capable implementaitons.")

    for param_type in TEST_PARAMS.keys():
        parser.add_argument(f"--{param_type}", type=int, default=0, help=f"Count for param type {param_type}")

    parser.add_argument(f"--test-all-count", type=int, default=0, help=f"Apply m inimumcount for all param types.")

    parser.add_argument("--accelerator", type=str, default="cuda", help="Device to use.")
    parser.add_argument("--file", type=str, default="speed_tests.json", help="Where to store results.")
    parser.add_argument("--overwrite", default=False, action="store_true", help="Whether to overwrite.")

    args = parser.parse_args()
    return args


def main(args=None):
    if args is None:
        args = parse_args()

    default = args.test_all_count
    if default > 0:
        for k in TEST_PARAMS:
            setattr(args, k, default)

    test_spec = {k: count for k in TEST_PARAMS if (count := getattr(args, k, 0)) > 0}
    
    print("Testing specification:")
    print(test_spec)
    results = {}

    implementations = args.implementations

    path = pathlib.Path(args.file)

    if args.all_gpu:
        implementations = ["sparse", "numba", "cupy", "triton"]
    if args.all_cpu:
        implementations = ["sparse", "pytorch", "numba"]

    if args.all_impl:
        implementations = MessagePassingKernels.get_available_implementations()
    if args.all_hidden:
        implementations = MessagePassingKernels.get_available_implementations(hidden=True)
    

    print("Testing implementations:")
    print(implementations)


    # Error if implementation does not exist.
    for impl in implementations:
        MessagePassingKernels.get_implementation(impl)

    if path.suffix != ".json":
        raise AssertionError(f"File extension not allowed! Suffix: '{path.suffix}'")

    if not args.overwrite:
        if path.exists():
            raise FileExistsError(f"Will not overwrite existing file! {path}")

    if len(implementations) == 0:
        raise ValueError("nothing to test")
    print("Testing implementations:", implementations)
    for impl in implementations:
        print("Testing implementation:", impl)
        tester = EnvOpsTester(impl)
        results[impl] = impl_results = {}
        for k, count in test_spec.items():
            print(f"Testing {k} {count} times:")
            np.random.seed(args.seed)
            try:
                out0, out1 = tester.check_speed(
                n_repetitions=count, device=torch.device(args.accelerator), data_size=TEST_PARAMS[k], compare_against=impl)
                impl_results[k] = dict(tested=out0, comparison=out1)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as toom:
                print(toom)
                print("Likely got out of memory for this test! Attempting to continue.")
                impl_results[k] = "OUT OF MEMORY"
            except Exception as ee:
                print(ee)
                print("Unknown error, but attempting to continue")
                impl_results[k] = "Unknown Error"

    with open(path, "wt") as f:
        json.dump(results, f)

    return results


if __name__ == "__main__":
    main()
