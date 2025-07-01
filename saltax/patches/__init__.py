from packaging import version
import straxen

if version.parse(straxen.__version__.split("-")[0]) < version.parse("3.0.0"):
    from . import straxen_2
else:
    from . import straxen_3
