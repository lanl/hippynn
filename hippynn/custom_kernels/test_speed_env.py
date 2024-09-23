import json
import pathlib

import numpy as np
import torch
from .autograd_wrapper import MessagePassingKernels

from .test_env import TEST_PARAMS, EnvOpsTester

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("implementations", type=str, nargs="*", help="Implementation(s) to test.")
    parser.add_argument("--seed", type=int, default=0, help="Seed")

    parser.add_argument("--all-hidden", action='store_true', default=False, help="Use all implementations, even with _ beginning.")
    parser.add_argument("--all-impl", action='store_true', default=False, help="Use all non-hidden implementations.")
    parser.add_argument("--all-gpu", action='store_true', default=False, help="Use low-mem implementations suitable for GPU.")

    for param_type in TEST_PARAMS.keys():
        parser.add_argument(
            f"--{param_type}",
            type=int,
            default=0,
            help=f"Count for param type {param_type}")

    parser.add_argument("--accelerator", type=str, default="cuda", help="Device to use.")
    parser.add_argument("--file", type=str, default="speed_tests.json", help="Where to store results.")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Whether to overwrite.")


    args = parser.parse_args()
    return args


def main(args=None):
    if args is None:
        args = parse_args()

    test_spec = {k: count for k in TEST_PARAMS if (count := getattr(args, k, 0)) > 0}
    results = {}

    implementations = args.implementations

    path = pathlib.Path(args.file)

    if path.suffix != ".json":
        raise AssertionError(f"File extension not allowed! {path.suffix}")

    if not args.overwrite:
        if path.exists():
            raise FileExistsError(f"Will not overwrite existing file! {path}")

    if args.all_gpu:
        implementations = ["sparse", "numba", "cupy", "triton"]
    if args.all_impl:
        implementations = MessagePassingKernels.get_available_implementations()
    if args.all_hidden:
        implementations = MessagePassingKernels._registered_implementations

    for impl in implementations:
        MessagePassingKernels.get_implementation(impl) # Error if does not exist

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
            out0, out1 = tester.check_speed(
                n_repetitions=count,
                device=torch.device(args.accelerator),
                data_size=TEST_PARAMS[k],
                compare_against=impl)
            impl_results[k] = dict(tested=out0, comparison=out1)

    with open(path, 'wt') as f:
        json.dump(results, f)

    return results


if __name__ == "__main__":
    main()

