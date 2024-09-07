"""
Settings to use in the rest of the library.
"""

# Setup defaults.
# Note for developers: Settings should not affect
# the semantics of the models or training.
# The custom kernels may have very slight effects
# due to numerical differences.
import warnings
import os
import configparser

from distutils.util import strtobool
from types import SimpleNamespace
from functools import partial


try:
    from tqdm.contrib import tqdm_auto

    TQDM_PROGRESS = tqdm_auto
except ImportError:
    try:
        from tqdm import tqdm

        TQDM_PROGRESS = tqdm
    except ImportError:
        TQDM_PROGRESS = None

if TQDM_PROGRESS is not None:
    DEFAULT_PROGRESS = partial(TQDM_PROGRESS, mininterval=1.0, leave=False)
else:
    DEFAULT_PROGRESS = None
### Progress handlers

def progress_handler(prog_str):
    if prog_str == "tqdm":
        return DEFAULT_PROGRESS
    elif prog_str.lower() == "none":
        return None
    else:
        try:
            prog_float = float(prog_str)
            return partial(TQDM_PROGRESS, mininterval=prog_float, leave=False)
        except:
            pass
    warnings.warn(f"Unrecognized progress setting: '{prog_str}'. Setting to none.")


def kernel_handler(kernel_string):

    kernel_string = kernel_string.lower()

    kernel = {
        "0": False,
        "false": False,
        "pytorch": False,
        "1": True,
        "true": True,
    }.get(kernel_string, kernel_string)

    if kernel not in [True, False, "auto", "triton", "cupy", "numba"]:
        warnings.warn(f"Unrecognized custom kernel option: {kernel_string}. Setting custom kernels to 'auto'")
        kernel = "auto"

    return kernel


# keys: defaults, types, and handlers
default_settings = {
    "PROGRESS": (DEFAULT_PROGRESS, progress_handler),
    "DEFAULT_PLOT_FILETYPE": (".pdf", str),
    "TRANSPARENT_PLOT": (False, strtobool),
    "DEBUG_LOSS_BROADCAST": (False, strtobool),
    "DEBUG_GRAPH_EXECUTION": (False, strtobool),
    "DEBUG_NODE_CREATION": (False, strtobool),
    "DEBUG_AUTOINDEXING": (False, strtobool),
    "USE_CUSTOM_KERNELS": ("auto", kernel_handler),
    "WARN_LOW_DISTANCES": (True, strtobool),
    "TIMEPLOT_AUTOSCALING": (True, strtobool),
    "PYTORCH_GPU_MEM_FRAC": (1.0, float),
}

settings = SimpleNamespace(**{k: default for k, (default, handler) in default_settings.items()})
settings.__doc__ = """
Values for the current hippynn settings.
See :doc:`/user_guide/settings` for a description.
"""

config_sources = {}  # Dictionary of configuration variable sources mapping to dictionary of configuration.
# We add to this dictionary in order of application

SECTION_NAME = "GLOBALS"

rc_name = os.path.expanduser("~/.hippynnrc")
if os.path.exists(rc_name) and os.path.isfile(rc_name):
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(rc_name)
    if SECTION_NAME not in config:
        warnings.warn(f"Config file {rc_name} does not contain a {SECTION_NAME} section and will be ignored!")
    else:
        config_sources["~/.hippynnrc"] = config[SECTION_NAME]

SETTING_PREFIX = "HIPPYNN_"
hippynn_environment_variables = {
    k.replace(SETTING_PREFIX, ""): v for k, v in os.environ.items() if k.startswith(SETTING_PREFIX)
}

LOCAL_RC_FILE_KEY = "LOCAL_RC_FILE"

if LOCAL_RC_FILE_KEY in hippynn_environment_variables:
    local_rc_fname = hippynn_environment_variables.pop(LOCAL_RC_FILE_KEY)
    if os.path.exists(local_rc_fname) and os.path.isfile(local_rc_fname):
        local_config = configparser.ConfigParser()
        local_config.read(local_rc_fname)
        if SECTION_NAME not in local_config:
            warnings.warn(f"Config file {local_rc_fname} does not contain a {SECTION_NAME} section and will be ignored!")
        else:
            config_sources[LOCAL_RC_FILE_KEY] = local_config[SECTION_NAME]
    else:
        warnings.warn(f"Local configuration file {local_rc_fname} not found.")

config_sources["environment variables"] = hippynn_environment_variables

for sname, source in config_sources.items():
    for key, value in source.items():
        key = key.upper()
        if key in default_settings:
            default, handler = default_settings[key]
            try:
                setattr(settings, key, handler(value))
            except Exception as ee:
                raise ValueError(f"Value {value} for setting {key} is invalid") from ee
        else:
            warnings.warn(f"Configuration source {sname} contains invalid variables ({key}). They will not be used.")
