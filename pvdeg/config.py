""" Configuration file for pvdeg package
"""

from pathlib import Path

#Specify module directories
PVDEG_DIR = Path(__file__).parent
REPO_NAME = __name__
DATA_DIR = PVDEG_DIR / "data"
TEST_DIR = PVDEG_DIR.parent / "tests"
TEST_DATA_DIR = PVDEG_DIR.parent / "tests" / "data"
MATERIALS_DIR = PVDEG_DIR.parent / "materials"
