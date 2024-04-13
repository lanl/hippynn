"""
Misc helpful functions which are not part of the library organization per se.
"""
import sys, os, traceback
import collections
import contextlib

import numpy as np
import torch

from . import settings


class teed_file_output:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, *arg, **kwargs):
        for stream in self.streams:
            stream.write(*arg, **kwargs)

    def flush(self, *arg, **kwargs):
        for stream in self.streams:
            stream.flush(*arg, **kwargs)


@contextlib.contextmanager
def log_terminal(file, *args, **kwargs):
    """
    :param: file: filename or string
    :param: args:   piped to ``open(file,*args,**kwargs)`` if file is a string
    :param: kwargs: piped to ``open(file,*args,**kwargs)`` if file is a string

    Context manager where stdout and stderr are redirected to the specified file
    in addition to the usual stdout and stderr.
    The manager yields the file. Writes to the opened file object with
    "with log_terminal(...) as <file>" will not automatically
    be piped into the terminal.
    """
    if getattr(file, "write", None) is None:
        close_on_exit = True
        file = open(file, *args, **kwargs)
    else:
        close_on_exit = False
    teed_stderr = teed_file_output(file, sys.stderr)
    teed_stdout = teed_file_output(file, sys.stdout)
    with contextlib.redirect_stderr(teed_stderr):
        with contextlib.redirect_stdout(teed_stdout):
            try:
                yield file
            except:
                print("Uncaught exception, logging interrupted.", file=teed_stderr)
                traceback.print_exc(file=file)
                raise
            finally:
                if close_on_exit:
                    file.close()


@contextlib.contextmanager
def active_directory(dirname, create=None):
    """
    Context manager for temporarily switching the current working directory.

    If create is None, always succeed.
    If create is True, only succeed if the directory does not exist, and create one.
    If create is False, only succeed if the directory does exist, and switch to it.

    In other words, use create=True if you want to force that it's a new directory.
    Use create=False if you want to switch to an existing directory.
    Use create=None create a directory if you are okay with either alternative.

    :param dirname: directory to enter
    :param create: (None,True,False)

    :return: None
    :raises: If directory status not compatible with `create` constraints.
    """
    if create not in (None, True, False):
        raise ValueError("Argument 'create' must be one of None, True, or False")

    exists = os.path.exists(dirname)

    if create is True and exists:
        raise FileExistsError(
            f"Directory '{dirname}' exists already. Pass create=False or create=None to "
            "allow switching to an existing directory."
        )
    if create is False and not exists:
        raise FileNotFoundError(
            f"Directory '{dirname}' not found. Pass create=True or create=None to allow creation of the directory."
        )
    if not exists and create in (None, True):
        os.makedirs(dirname)

    return_dir = os.getcwd()
    os.chdir(dirname)
    try:
        yield
    finally:
        os.chdir(return_dir)


def progress_bar(iterable, *args, **kwargs):
    if settings.PROGRESS is None:
        return iterable
    else:
        return settings.PROGRESS(iterable, *args, **kwargs)


def param_print(module):
    count = 0
    for pname, p in module.named_parameters():
        count += p.numel()
        req_grad = "Learned" if p.requires_grad else "Fixed"
        print(p.device, req_grad, p.dtype, p.shape, pname)
    print("Total Count:", count)


def device_fallback():
    device = (torch.cuda.is_available() and torch.device(torch.cuda.current_device())) or torch.device("cpu")
    print("Device was not specified. Attempting to default to device:", device)
    return device


def arrdict_len(array_dictionary):
    """
    Return the length of one of the arrays in a dictionary. Under the assumption that they are all the same.
    :param array_dictionary:
    :return:
    """
    return len(next(iter(array_dictionary.values())))


def print_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        print("Learning rate:{:>10.5g}".format(param_group["lr"]))


def isiterable(obj):
    return isinstance(obj, collections.abc.Iterable)


def pad_np_array_to_length_with_zeros(array, length, axis=0):
    n = array.shape[axis]
    m = length - n
    if m < 0:
        raise ValueError(f"Cannot pad array to negative length! Array length: {n}, Total length requested: {length}")
    pad_width = [[0, 0] for _ in array.shape]
    pad_width[axis][1] = m
    return np.pad(array, pad_width, mode="constant")


def np_of_torchdefaultdtype():
    return torch.ones(1, dtype=torch.get_default_dtype()).numpy().dtype

def is_equal_state_dict(d1, d2):
    """
    Checks if two pytorch state dictionaries are equal. Calls itself recursively
    if the value for a parameter is a dictionary.


    :param d1:
    :param d2:
    :return:
    """
    if set(d1.keys()) != set(d2.keys()):
        # They have different sets of keys.
        return False
    for k in d1:
        v1 = d1[k]
        v2 = d2[k]
        if type(v1) != type(v2):
            return False
        if isinstance(v1, torch.Tensor):
            if torch.equal(v1, v2):
                continue
            else:
                return False
        elif isinstance(v1, dict):
            # call recursive:
            return is_equal_state_dict(v1, v2)
        elif v1 != v2:
            return False

    return True


