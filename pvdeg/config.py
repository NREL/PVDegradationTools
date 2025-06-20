"""Configuration file for pvdeg package."""

from pathlib import Path

# Specify module directories
PVDEG_DIR = Path(__file__).parent
REPO_NAME = __name__
DATA_DIR = PVDEG_DIR / "data"
TEST_DIR = PVDEG_DIR.parent / "tests"
TEST_DATA_DIR = PVDEG_DIR.parent / "tests" / "data"

# downloader target directory
METOROLOGICAL_DOWNLOAD_PATH = Path.home() / "PVDeg-Meteorological"

# DATA_LIBRARY = PVDEG_DIR.parent / "DataLibrary"
# if not os.path.isdir(DATA_LIBRARY):
#     DATA_LIBRARY = os.path.join(sys.prefix, "DataLibrary")
#     if not os.path.isdir(DATA_LIBRARY):
#         print("DataLibrary not found in {DATA_LIBRARY} or {PVDEG_DIR.parent}.")
