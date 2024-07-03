from importlib.metadata import version
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
from . import montecarlo
from .scenario import Scenario
from . import spectral
from . import standards
from . import temperature
from . import utilities
from . import weather

__version__ = version("pvdeg")

# default python logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel("DEBUG")


# numba_logger = logging.getLogger('numba')
# numba_logger.setLevel(logging.WARNING)
# numba_console_handler = logging.StreamHandler()
# numba_console_handler.setLevel(logging.DEBUG)
# numba_console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# numba_console_handler.setFormatter(numba_console_formatter)
# numba_logger.addHandler(numba_console_handler)