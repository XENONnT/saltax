from ..utils import straxen_version

if straxen_version() == 2:
    from . import straxen_2
elif straxen_version() == 3:
    from . import straxen_3
