#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    raise RuntimeError('setuptools is required')

import versioneer
from glob import glob

DESCRIPTION = ('Pvdeg is a python library that supports the calculation of' +
               'degradation related parameters for photovoltaic (PV) modules.')

with open('README.md', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

DISTNAME = 'pvdeg'
LICENSE = 'BSD-3'
AUTHOR = 'Pvdeg Python Developers'
AUTHOR_EMAIL = 'Michael.Kempe@nrel.gov'
MAINTAINER_EMAIL = 'Silvana.Ovaitt@nrel.gov'
URL = 'https://github.com/NREL/PVDegradationTools'

PACKAGES = ['pvdeg']

KEYWORDS = [
    'photovoltaic',
    'solar',
    'degradation',
    'analysis',
    'performance',
    'module',
    'PV'
]

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering'
]

PROJECT_URLS = {
    "Bug Tracker": "https://github.com/NREL/PVDegradationTools/issues",
    "Documentation": "https://pvdegradationtools.readthedocs.io/",
    "Source Code": "https://github.com/NREL/PVDegradationTools",
}

PYTHON_REQUIRES = '>=3.9.0'

SETUP_REQUIRES = [
    'pytest-runner',
]

DOCS_REQUIRE = [
    'sphinx',
    'sphinx_rtd_theme'
]

TESTS_REQUIRE = [
    'pytest',
    'coverage'
]

EXTRAS_REQUIRE = {
    'docs': DOCS_REQUIRE,
    'test': TESTS_REQUIRE,
}

EXTRAS_REQUIRE['all'] = sorted(set(sum(EXTRAS_REQUIRE.values(), [])))

with open("requirements.txt") as f:
    INSTALL_REQUIRES = f.read().splitlines()

DATA_FILES = [('DataLibrary', glob('DataLibrary/*'))]

ENTRY_POINTS={"console_scripts": ["pvdeg=pvdeg.cli:cli"]}

setuptools_kwargs = {
    'zip_safe': False,
    'scripts': [],
    'include_package_data': True,
    'package_dir' : {"pvdeg": "pvdeg"},
}

setup(name=DISTNAME,
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      python_requires=PYTHON_REQUIRES,
      packages=PACKAGES,
      keywords=KEYWORDS,
      #setup_requires=SETUP_REQUIRES,
      tests_require=TESTS_REQUIRE,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      project_urls=PROJECT_URLS,
      classifiers=CLASSIFIERS,
      data_files=DATA_FILES,
      entry_points=ENTRY_POINTS,
      **setuptools_kwargs)