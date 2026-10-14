<img src="https://raw.githubusercontent.com/NatLabRockies/PVDegradationTools/refs/heads/main/docs/source/_static/logo-vectors/PVdeg-Logo-Horiz-Color.svg" width="600">


<table>
<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/NatLabRockies/PVDegradationTools/blob/main/LICENSE.md">
    <img src="https://img.shields.io/pypi/l/pvlib.svg" alt="license" />
    </a>
</td>
</tr>
<tr>
  <td>Publications</td>
  <td>
     <a href="https://doi.org/10.5281/zenodo.8088382"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8088382.svg" alt="DOI"></a>
  </td>
</tr>
<tr>
  <td>Documentation</td>
  <td>
	<a href='https://PVDegradationTools.readthedocs.io'>
	    <img src='https://readthedocs.org/projects/pvdegradationtools/badge/?version=latest' alt='Documentation Status' />
	</a>
  </td>
</tr>
<tr>
  <td>Build status</td>
  <td>
   <a href="https://github.com/NatLabRockies/PVDegradationTools/actions/workflows/pytest.yml?query=branch%3Amain">
      <img src="https://github.com/NatLabRockies/PVDegradationTools/actions/workflows/pytest.yml/badge.svg?branch=main" alt="GitHub Actions Testing Status" />
   </a>
   <a href="https://codecov.io/gh/NatLabRockies/PVDegradationTools" >
   <img src="https://codecov.io/gh/NatLabRockies/PVDegradationTools/graph/badge.svg?token=4I24S8BTG7"/>
   </a>
  </td>
</tr>
</table>



# PVDeg: Python Package for Modeling Degradation of Photovoltaic Systems

PVDeg is an open-source Python package for modeling photovoltaic (PV)
degradation, developed at the National Laboratory of the Rockies
(NLR, formerly NREL) and supported by the Durable Module Materials (DuraMAT)
consortium. It provides modular functions, materials databases, and calculation
workflows for simulating degradation mechanisms (e.g., LeTID, hydrolysis, UV
exposure) using weather data from the National Solar Radiation Database (NSRDB)
and the Photovoltaic Geographical Information System (PVGIS). By integrating
Monte Carlo uncertainty propagation and geospatial processing, PVDeg enables
field-relevant predictions and uncertainty quantification of module reliability
and lifetime.

## Key Features

- **Core Degradation Functions**: Dedicated functions for physical degradation mechanisms including moisture ingress, LeTID, UV exposure, and thermal stress
- **Scenario Class**: Simplified workflow interface for complex multi-parameter degradation studies
- **Geospatial Analysis**: Large-scale spatial analyses with parallel processing across geographic regions
- **Monte Carlo Framework**: Uncertainty quantification through parameter distribution sampling
- **Material Databases**: Curated degradation parameters, kinetic coefficients, and material properties
- **Weather Data Integration**: Seamless access to NSRDB and PVGIS meteorological data
- **Standards Support**: Contributions to IEC TS 63126 and other standardization efforts
## Example Applications

PVDeg has been adopted in multiple studies across the PV reliability community:

- **Thermal Stability and IEC TS 63126 Compliance**: Calculate effective standoff distances and generate public maps supporting the IEC TS 63126 standard
- **Light and Elevated Temperature Induced Degradation (LeTID)**: Integrated into international interlaboratory comparison studies and field-aged array analyses
- **Geospatial Performance Modeling**: Coupled with GeoGridFusion to streamline weather-data storage and spatial queries for large-scale degradation simulations
- **Agrivoltaics and System-Level Modeling**: Combined with PySAM to assess degradation-driven yield losses in dual-use agrivoltaic systems
- **Material-Property Parameterization**: Studies of UV-induced polymer degradation and moisture-related failures in encapsulants and backsheets

Documentation
=============

Full documentation is available at **[ReadTheDocs](https://pvdegradationtools.readthedocs.io)** including:

- 📚 [API Reference](https://pvdegradationtools.readthedocs.io/en/latest/api.html) - Complete function and class documentation
- 📖 [User Guide](https://pvdegradationtools.readthedocs.io/en/latest/user_guide/installation.html) - Installation, tutorials, and usage guides
- 🔧 [Contributing Guide](https://pvdegradationtools.readthedocs.io/en/latest/user_guide/contributing.html) - Development setup and guidelines
- 📰 [What's New](https://pvdegradationtools.readthedocs.io/en/latest/whatsnew/index.html) - Release notes and changelogs

Installation
============

PVDeg releases may be installed using `pip` and `conda` tools. Compatible with Python 3.10 and above.

**Quick Install:**

```bash
pip install pvdeg
```

**With optional dependencies:**

| Group | Install Command | Purpose |
|-------|----------------|---------|
| **sam** | `pip install pvdeg[sam]` | PySAM support for system modeling |
| **docs** | `pip install pvdeg[docs]` | Sphinx documentation tools |
| **test** | `pip install pvdeg[test]` | Testing and validation tools |
| **books** | `pip install pvdeg[books]` | Jupyter Book publishing |
| **all** | `pip install pvdeg[all]` | All optional dependencies |

**Developer Installation:**

```bash
git clone https://github.com/NatLabRockies/PVDegradationTools.git
cd PVDegradationTools
pip install -e .[all]
```

📖 **For detailed installation instructions including conda environments, HPC setup, troubleshooting, and version compatibility, see the [Installation Guide](https://pvdegradationtools.readthedocs.io/en/latest/user_guide/installation.html).**

Tutorials
=========

PVDeg provides comprehensive tutorials organized by topic. Choose your preferred environment:

### Jupyter Book (Recommended)

Interactive tutorials with live execution: [**PVDeg Jupyter Book**](https://NatLabRockies.github.io/PVDegradationTools/intro.html)

- Click the 🚀 rocket icon to launch notebooks in [Google Colab](https://colab.research.google.com/)
- **Development Preview:** See latest changes at [dev-preview](https://NatLabRockies.github.io/PVDegradationTools/dev-preview/intro.html)

### Binder

Run tutorials in your browser without installation: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NatLabRockies/PVDegradationTools/main)

### Local Installation

1. **Install PVDeg** (see [Installation](#installation))

2. **Clone the repository** to access tutorial notebooks:
   ```bash
   git clone https://github.com/NatLabRockies/PVDegradationTools.git
   cd PVDegradationTools
   ```

3. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

4. **Navigate to tutorials** organized by category:
   - `01_basics/` - Introduction to PVDeg fundamentals
   - `02_degradation/` - Degradation mechanism modeling
   - `03_monte_carlo/` - Monte Carlo uncertainty analysis
   - `04_scenario/` - Scenario-based modeling workflows
   - `05_geospatial/` - Geospatial and HPC scenarios
   - `06_advanced/` - Advanced topics and API access
   - `10_workshop_demos/` - Workshop demonstrations
   - `tools/` - Standalone analysis tools

📖 **For more information on running and validating notebooks, see the [documentation](https://pvdegradationtools.readthedocs.io).**

License
=======

[BSD 3-clause](https://github.com/NatLabRockies/PVDegradationTools/blob/main/LICENSE.md)


Contributing
============

We welcome contributions to this software! Please see
[CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions on:
- Setting up your development environment
- Installing and using pre-commit hooks
- Code style guidelines
- Testing and documentation requirements
- **Contributing to material property and degradation parameter databases**
- Submitting pull requests

Please read the copyright license agreement (cla-1.0.md), with instructions on
signing it in sign-CLA.md.

All code, documentation, and discussion contributors are acknowledged for their
contributions to the PVDeg project.


Getting support
===============

If you suspect that you may have discovered a bug or if you'd like to
change something about PVDeg, then please make an issue on our
[GitHub issues page](https://github.com/NatLabRockies/PVDegradationTools/issues).


Citing
======

If you use PVDeg in a published work, please cite both the software and the paper.

**Software Citation:**

	Springer, M., Ovaitt, S., Daxini, R., Ford, T., Brown, M., Karas, J., Holsapple, D., & Kempe, M. (2026). PVDeg: a Python package for modeling degradation of solar photovoltaic systems [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.8088382

```bibtex
@software{pvdeg,
  author    = {Springer, Martin and Ovaitt, Silvana and Daxini, Rajiv and
               Ford, Tobin and Brown, Matthew and Karas, Joseph and
               Holsapple, Derek and Kempe, Michael},
  title     = {{PVDeg: a Python package for modeling degradation of solar
               photovoltaic systems}},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.8088382},
  url       = {https://doi.org/10.5281/zenodo.8088382}
}
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8088382.svg)](https://doi.org/10.5281/zenodo.8088382)

The DOI above always resolves to the latest release. To cite the exact version you used, pick it from the version list on the Zenodo page, or use the **"Cite this repository"** button in the right sidebar.

**JOSS Paper (In Review):**

	Daxini, R., Ovaitt, S., Springer, M., Ford, T., & Kempe, M. (2026). PVDeg: a Python package for modeling degradation of solar photovoltaic systems. Journal of Open Source Software (In Review).


Acknowledgements
================

This work was authored in part by the National Laboratory of the Rockies
for the U.S. Department of Energy (DOE), operated under Contract No.
DE-AC36-08GO28308. Funding provided as part of the Durable Module Materials
Consortium 2 (DuraMAT 2) funded by the by U.S. DOE Office of Critical Minerals
and Energy Innovation (CMEI) Solar Energy Technologies Office (SETO), Agreement
38259. The views expressed in the article do not necessarily represent the
views of the DOE or the U.S. Government. The U.S. Government retains and the
publisher, by accepting the article for publication, acknowledges that the U.S.
Government retains a nonexclusive, paid-up, irrevocable, worldwide license to
publish or reproduce the published form of this work, or allow others to do so,
for U.S. Government purposes

HPC disclaimer: A portion of This research was performed using computational
resources sponsored by the U.S. Department of Energy's Office of Critical
Minerals and Energy Innovation and located at the National Laboratory of the
Rockies.

**NLR Software Record:** SWR-20-71 (Holsapple, Derek; Ayala Pelaez, Silvana; Kempe, Michael. "PV Degradation Tools", NLR Github 2020)
