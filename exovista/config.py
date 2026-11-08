import os
import warnings
from types import SimpleNamespace


def env_boolean(var, default=None):
    value = os.getenv(var, default)
    if value is None:
        return default
    if value.lower() in ("false", "0", "off", ""):
        return False
    return True


def env_int(var, default):
    value = os.getenv(var)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_choice(var, default, choices):
    """Integer environment setting in ``choices``; warns about other values."""
    value = os.getenv(var)
    if value is None or value.strip() == "":
        return default
    try:
        number = int(value)
    except ValueError:
        number = None
    if number not in choices:
        warnings.warn(
            f"ignoring {var}={value!r}: expected one of "
            f"{_describe(choices)}; using {default}",
            UserWarning,
            stacklevel=3,
        )
        return default
    return number


def _describe(choices):
    choices = sorted(choices)
    if choices == list(range(choices[0], choices[-1] + 1)) and len(choices) > 2:
        return f"the integers {choices[0]} to {choices[-1]}"
    return " or ".join(str(c) for c in choices)


def initialize_config():
    cfg = SimpleNamespace()
    cfg.debug = env_boolean("EXODUSII_DEBUG", default="off")
    # zlib/deflate level applied to array variables when writing netCDF4 files.
    # 1-9 enables compression (1 = fast, captures most of the savings; 9 = max).
    # 0 disables compression (legacy behavior). Other values are ignored with a
    # warning, like an invalid compression_level= raises.
    cfg.compression_level = env_choice("EXODUSII_COMPRESSION_LEVEL", 1, range(10))
    # 8 (float64, SEACAS default) or 4 (float32, exovista < 2 behaviour)
    cfg.float_word_size = env_choice("EXODUSII_FLOAT_WORD_SIZE", 8, (4, 8))
    return cfg


config = initialize_config()
