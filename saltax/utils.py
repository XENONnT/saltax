def replace_source(src, olds, news):
    assert len(olds) == len(news), "Number of old and new codes must match"
    for old, new in zip(olds, news):
        assert old in src, f"Codes {old} not found in source"
        src = src.replace(old, new)
    return src
