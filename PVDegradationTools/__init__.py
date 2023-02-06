#!/usr/bin/env python3

"""Collection of submodules of PVDegradationTools.
"""

from .main import StressFactors, Degradation, BOLIDLeTID

from . import standards
from . import _version

__version__ = _version.get_versions()['version']
