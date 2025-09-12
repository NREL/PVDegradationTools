from importlib.metadata import version
import logging

from .config import *

# from . import cli
from . import collection
from . import decorators
from . import degradation
from . import design
from . import fatigue
from . import geospatial
#from .geospatialscenario import GeospatialScenario
from . import humidity
from . import letid
from . import montecarlo
from . import pysam
from .scenario import Scenario, GeospatialScenario
from . import spectral
from . import store
from . import symbolic
from . import standards
from . import temperature
from . import utilities
from . import weather
from . import diffusion

__version__ = version("pvdeg")

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel("DEBUG")
