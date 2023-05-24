import logging
from pathlib import Path
import sys

#TODO: Delete once all functions are split up into separate files
from .main import StressFactors, Degradation, Scenario

from . import cli
from . import collection
from . import degradation
from . import design
from . import fatigue
from . import humidity
from . import letid
from .scenario import Scenario
from . import spectral
from . import standards
from . import temperature
from . import utilities
from . import weather
from . import _version

__version__ = _version.get_versions()['version']

PVDEG_DIR = Path(__file__).parent
REPO_NAME = __name__
DATA_DIR = PVDEG_DIR / "data"
TEST_DATA_DIR = PVDEG_DIR.parent / "tests" / "data"
MATERIALS_DIR = PVDEG_DIR.parent / "materials"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel("DEBUG")
