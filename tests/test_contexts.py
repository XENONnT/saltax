import saltax


RUNID = 53169


def test_import_cutax():
    import saltax.contexts

def test_contexts_import():
    from saltax.contexts import sxenonnt

def test_contexts_definition():
    from saltax.contexts import SALTAX_MODES
    for saltax_mode in SALTAX_MODES:
        if saltax_mode == 'salt':
            runid == RUNID
        else:
            runid = None
        st = saltax.contexts.sxenonnt(runid=runid,
                                      saltax_mode=saltax_mode)
        assert st is not None, "sxenonnt should be defined for saltax_mode %s"%(saltax_mode)
