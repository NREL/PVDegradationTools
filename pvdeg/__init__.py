"""init.py."""

from importlib.metadata import version
import logging

from .config import *  # noqa: F403

# from . import cli
from . import collection
from . import decorators
from . import degradation
from . import design
from . import fatigue
from . import geospatial
from . import humidity
from . import letid
from . import montecarlo
from . import pysam
from . import spectral
from . import symbolic
from . import standards
from . import temperature
from . import utilities
from . import weather
from . import diffusion

from .scenario import Scenario
from .geospatialscenario import GeospatialScenario

__version__ = version("pvdeg")

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel("DEBUG")
