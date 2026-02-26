import sys
from packaging import version
from utilix import xent_collection
import straxen

COLL = xent_collection()


def straxen_version():
    """Get the major version of straxen.

    :return: Major version of straxen as an integer.

    """
    v = version.parse(straxen.__version__.split("-")[0])
    if v < version.parse("3.0.0"):
        return 2
    elif v >= version.parse("3.0.0") and v < version.parse("4.0.0"):
        return 3
    else:
        raise ValueError(
            f"Unsupported straxen version: {straxen.__version__}. "
            "Please use straxen version 2.x or 3.x."
        )


def replace_source(src, olds, news):
    """Replace old codes with new codes in the source string.

    :param src: Source code as a string.
    :param olds: List of old codes to be replaced.
    :param news: List of new codes to replace the old ones.
    :return: Modified source code with old codes replaced by new codes.

    """
    assert len(olds) == len(news), "Number of old and new codes must match"
    for old, new in zip(olds, news):
        assert old in src, f"Codes {old} not found in source"
        src = src.replace(old, new)
    return src


def setattr_module(mod: str, name: str, value):
    """Set an attribute in a module by its name.

    Setting the attribute in the module's namespace in all depth to make sure all possible
    reimported namesapce got overwritten.
    :param mod: The module name as a string.
    :param name: The name of the attribute to set.
    :param value: The value to set the attribute to.

    """

    parts = mod.split(".")
    for depth in range(1, len(parts) + 1):
        _name = ".".join(parts[:depth])
        # assuming the module is already imported
        _mod = sys.modules.get(_name)
        setattr(_mod, name, value)
