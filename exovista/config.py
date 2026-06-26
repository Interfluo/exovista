import os
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


def initialize_config():
    cfg = SimpleNamespace()
    cfg.use_netcdf4_if_possible = env_boolean("EXODUSII_USE_NETCDF4", default="on")
    cfg.debug = env_boolean("EXODUSII_DEBUG", default="off")
    # zlib/deflate level applied to array variables when writing netCDF4 files.
    # 1-9 enables compression (1 = fast, captures most of the savings; 9 = max).
    # 0 disables compression (legacy behavior). Clamped to the valid range.
    cfg.compression_level = min(max(env_int("EXODUSII_COMPRESSION_LEVEL", 1), 0), 9)
    return cfg


config = initialize_config()
