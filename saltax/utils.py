import sys


def replace_source(src, olds, news):
    assert len(olds) == len(news), "Number of old and new codes must match"
    for old, new in zip(olds, news):
        assert old in src, f"Codes {old} not found in source"
        src = src.replace(old, new)
    return src


def setattr_module(mod: str, name: str, value):
    """Set an attribute in a module by its name."""

    parts = mod.split(".")
    for depth in range(1, len(parts) + 1):
        _name = ".".join(parts[:depth])
        # reuse existing module if already imported
        _mod = sys.modules.get(_name)
        setattr(_mod, name, value)
