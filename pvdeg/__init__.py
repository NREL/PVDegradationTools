import logging

from .config import *

# from . import cli
from . import collection
from . import degradation
from . import design
from . import fatigue
from . import geospatial
from . import humidity
from . import letid
from .scenario import Scenario
from . import spectral
from . import standards
from . import temperature
from . import utilities
from . import weather
from . import _version

__version__ = _version.get_versions()["version"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel("DEBUG")
