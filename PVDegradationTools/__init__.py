import logging
from pathlib import Path

from .main import StressFactors, Degradation, BOLIDLeTID, Scenario

from . import cli
from . import humidity
from . import standards
from . import utilities
from . import _version

__version__ = _version.get_versions()['version']

PVD_DIR = Path(__file__).parent
REPO_NAME = __name__
TEST_DATA_DIR = PVD_DIR.parent / "tests" / "data"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel("DEBUG")