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
from typing import Union

from distutils.util import strtobool
from types import SimpleNamespace
from functools import partial

# Significant strings
SECTION_NAME = "GLOBALS"
SETTING_PREFIX = "HIPPYNN_"
LOCAL_RC_FILE_KEY = "LOCAL_RC_FILE"

# Globals
DEFAULT_PROGRESS = None  # this gets set to a tqdm object if possible
TQDM_PROGRESS = None  # the best progress bar from tqdm, if available.

def setup_tqdm():
    global TQDM_PROGRESS
    global DEFAULT_PROGRESS
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

# Setting handlers: Take an input str or other value and return the appropriate value.

def progress_handler(prog_setting: Union[str, float, bool, None]):
    """
    Function for handling the progress bar settings.

    :param prog_setting:
    :return:
    """
    if TQDM_PROGRESS is None:
        setup_tqdm()

    if prog_setting in (True, False, None):
        prog_setting = {
            True: "tqdm",
            False: "none",
            None: "none",
        }[prog_setting]

    if isinstance(prog_setting, str):
        prog_setting = prog_setting.lower()
        if prog_setting == "tqdm":
            return DEFAULT_PROGRESS
        elif prog_setting.lower() == "none":
            return None

    prog_setting = float(prog_setting)  # Trigger error if not floatable.

    return partial(TQDM_PROGRESS, mininterval=prog_setting, leave=False)



def kernel_handler(kernel_string):
    """

    :param kernel_string:
    :return:
    """
    kernel_string = kernel_string.lower()

    kernel = {
        "0": False,
        "false": False,
        "1": True,
        "true": True,
    }.get(kernel_string, kernel_string)

    # This function used to warn about unexpected kernel settings.
    # Now this is an error which is raised in the custom_kernels module.

    return kernel


def bool_or_strtobool(key: Union[bool, str]):
    if isinstance(key, bool):
        return key
    else:
        return strtobool(key)


# keys: defaults, types, and handlers.
DEFAULT_SETTINGS = {
    "PROGRESS": ('tqdm', progress_handler),
    "DEFAULT_PLOT_FILETYPE": (".pdf", str),
    "TRANSPARENT_PLOT": (False, bool_or_strtobool),
    "DEBUG_LOSS_BROADCAST": (False, bool_or_strtobool),
    "DEBUG_GRAPH_EXECUTION": (False, bool_or_strtobool),
    "DEBUG_NODE_CREATION": (False, bool_or_strtobool),
    "DEBUG_AUTOINDEXING": (False, bool_or_strtobool),
    "USE_CUSTOM_KERNELS": ("auto", kernel_handler),
    "WARN_LOW_DISTANCES": (True, bool_or_strtobool),
    "TIMEPLOT_AUTOSCALING": (True, bool_or_strtobool),
    "PYTORCH_GPU_MEM_FRAC": (1.0, float),
    "COMM_FEATURES_LAMMPS": (True, bool_or_strtobool),
}

INITIAL_SETTINGS = {k: handler(default) for k, (default, handler) in DEFAULT_SETTINGS.items()}

settings = SimpleNamespace(**INITIAL_SETTINGS)
settings.__doc__ = """
Values for the current hippynn settings.
See :doc:`/user_guide/settings` for a description.
"""


def reload_settings(**kwargs):
    """
    Attempt to reload the hippynn library settings.

    Settings sources are, in order from least to greatest priority:
        - Default values
        - The file `~/.hippynnrc`, which is a standard python config file which contains
           variables under the section name [GLOBALS].
        - A file specified by the environment variable `HIPPYNN_LOCAL_RC_FILE`
           which is treated the same as the user rc file.
        - Environment variables prefixed by ``HIPPYNN_``, e.g. ``HIPPYNN_DEFAULT_PLOT_FILETYPE``.
        - Keyword arguments passed to this function.


    :param kwargs: explicit settings to change.

    :return:
    """
    # Developer note: this function modifies the module-scope `settings` directly.

    config_sources = {}  # Dictionary of configuration variable sources mapping to dictionary of configuration.
    # We add to this dictionary in order of application

    rc_name = os.path.expanduser("~/.hippynnrc")
    if os.path.exists(rc_name) and os.path.isfile(rc_name):
        config = configparser.ConfigParser(inline_comment_prefixes="#")
        config.read(rc_name)
        if SECTION_NAME not in config:
            warnings.warn(f"Config file {rc_name} does not contain a {SECTION_NAME} section and will be ignored!")
        else:
            config_sources["~/.hippynnrc"] = config[SECTION_NAME]

    hippynn_environment_variables = {
        k.replace(SETTING_PREFIX, ""): v for k, v in os.environ.items() if k.startswith(SETTING_PREFIX)
    }

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
    config_sources["kwargs"] = kwargs.copy()

    for sname, source in config_sources.items():
        for key, value in source.items():
            key = key.upper()
            if key in DEFAULT_SETTINGS:
                default, handler = DEFAULT_SETTINGS[key]
                try:
                    setattr(settings, key, handler(value))
                except Exception as ee:
                    raise ValueError(f"Value {value} for setting {key} is invalid") from ee
            else:
                warnings.warn(f"Configuration source {sname} contains invalid variables ({key}). These will be ignored.")

    return settings


reload_settings()

