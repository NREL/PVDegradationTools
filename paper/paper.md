---
title: 'PVDeg: a python package for modeling degradation on solar photovoltaic systems'
tags:
  - Python
  - solar energy
  - photovoltaics
  - renewable energy
  - degradation mechanisms
authors:
  - name: Silvana Ovaitt
    orcid: 0000-0003-0180-728X
    affiliation: 1
  - name: Martin Springer
    orcid: 0000-0001-6803-108X
    affiliation: 1
  - name: Tobin Ford
    orcid: 0009-0000-7428-5625
    affiliation: 1
  - name: Rajiv Daxini 
    orcid: 0000-0003-1993-9408
    affiliation: 1
  - name: Michael Kempe 
    orcid: 0000-0003-3312-0482
    affiliation: 1
affiliations:
 - name: National Renewable Energy Laboratory (NREL)
   index: 1
date: 15 August 2025
bibliography: paper.bib
---

# Summary

[comment]: Brief Intro
PVDegradationTools, or PVDeg for short, is a national-laboratory-developed, community-supported, open-source toolkit that provides a set of functions and classes for simulating the degradation of photovoltaic (PV) systems. PVDeg automates calculations of PV system degradation and performance. Specific algorithms include .... , among others. PVDeg is an important component of a growing ecosystem of open-source tools for solar energy [@Holmgren2018].

[comment]: Image of PVDeg results or something?
![Visualization of a bifacial photovoltaic array generated through bifacial_radiance. Courtesy of J. Alderman.\label{fig:visualization}](Alderman.PNG){ width=80% }

[comment]: Hosting and documentation
PVDeg is hosted on Github and PyPi, and it was developed by contributors from national laboratories, academia, and private industry. PVDeg is copyrighted by the Alliance for Sustainable Energy with a BSD 3-clause license allowing permissive use with attribution. PVDeg is extensively tested for functional and algorithm consistency. Continuous integration services check each pull request on Linux and Python versions 2.7 and 3.6. PVDeg is thoroughly documented, and detailed tutorials are provided for many features. The documentation includes help for installation and guidelines for contributions. The documentation is hosted at readthedocs.org. Github’s issue trackers provide venues for user discussions and help.

[comment]: Introducing the 3 parts: classes/functions, the library, and the
geospatial
The PVDeg python library is a spatio-temporal modeling assessment tool
that empowers users to calculate various PV degradation modes, for different
PV technologies and materials in its database. It is designed to serve the PV
community, including researchers, device manufacturers, and other PV
stakeholders to assess different degradation modes in locations around the
world. The library is developed and hosted open-source on GitHub, and is
structured in three layers: core functions and classes, scenario analysis, and
geospatial analysis. These algorithms are typically implementations of models
published in the existing peer-reviewed literature. In addition, data for
PVDeg is sourced from the National Solar Radiation Database and the (NSRDB)
and Photovoltaic Geographcial Information System (PVGIS). The core API
consists of functions and classes [XX...]. The second layer is the scenario
analysis

The bifacial_radiance API and graphical user interface (GUI) were designed to serve the various needs of the many subfields of bifacial solar panel power research and engineering. The intended audience ranges from PV performance researchers, Engineering Procurement Construction (EPC) companies, installers, investors, consumers and analysts of the PV industry interested in predicting and evaluating bifacial photovoltaic systems. It is implemented in three layers: core RADIANCE-interface functions; ``Bifacial-Radiance``, ``Meteorological``, ``Scene``, and ``Analysis`` classes; and the ``GUI`` and ``model-chain`` classes. The core API consists of a collection of functions that implement commands directly to the RADIANCE software. These commands are typical implementations of algorithms and models described in peer-reviewed publications. The functions provide maximum user flexibility; however, some of the function arguments require an unwieldy number of parameters. The next API level contains the ``Bifacial-Radiance``, ``Meteorological``, ``Scene``, and ``Analysis`` classes. These abstractions provide simple methods that wrap the core function API layer and communicate with the RADIANCE software, which provides ray-trace processing capabilities. The method API simplification is achieved by separating the data that represent the object (object attributes) from the data that the object methods operate on (method arguments). For example, a ``Bifacial-Radiance`` object is represented by a ``module`` object, meteorological data, and ``scene`` objects. The ``gendaylit`` method operates on the meteorological data to calculate solar position with the support of algorithms from pvlib python [@pvlib], and generate corresponding sky files, linking them to the ``Bifacial-Radiance`` object. Then the ``makeOct`` method combines the sky files, ``module`` and ``scene`` objects when calling the function layer, returning the results from an ``Analysis`` object to the user. The final level of API is the ``ModelChain`` class, designed to simplify and standardize the process of stitching together the many modeling steps necessary to convert a time series of weather data to AC solar power generation, given a PV system and a location. The ``ModelChain`` also powers the ``GUI``, which provides a cohesive visualization of all the input parameters and options for most common modeling needs.

[comment]: Release Info
Holsapple, Derek, Ayala Pelaez, Silvana, Kempe, Michael. "PV Degradation Tools", NREL Github 2020, Software Record SWR-20-71.

PVDeg was first coded in Python and released as a stable version in Github after submission to the a U.S. Department of Energy Code project on 2020 [@Holsapple]. Expansions for geospatial and monte carlo framework were undertaken in 2024 [@REF]. Additional features continue to be added as described in @REF, and in the documentation’s “What’s New” section.

[comment]: LITERATURE USES  -- Mike kempe's standard, paper, pvdeg papers showcasin use, duramat webinars... agrivoltaics publication, stalled tracker analysis
PVDeg has been used in numerous studies, for example, in [@ ], for calculating effective standoff distances across the US following IEC 63126 for thermal stability, as well as on the standard itself, IEC TS 63126, for providing user-friendly maps for calculating these standoff distances for any worldwide location. [@IEC TS 6t3126]. It has also been used for studying the senstivity of diffuse-tracing methodologies geospatially to time reponse and weather, modeling the yearly performance and tracker movement change under smart tracking algorithms  in [@Adinolfi2024] and [@Adinolfi2025]. It's most recent use is also on the implementation of geospatial modeling of pySAM simulatins for claculation of ground-irradiance for dual-use agrivoltaics [@PVSC 2023], wiht pending dataset publication. an example of geospatial and geo-temporal results can also be seen in [@Karas] study of the LeTID process phenomenon, where users can observe the output of module degradaiton and improvement over a period of various years to these recently discovered and parameterized degradation.


[comment]: More detail in geospatial and monte carlo usefulness?
Perhaps the most power and area of focus in the PVDeg result, taht has warranted its uptake in the PV Reliability communiyt, are both the geospatial and monte carlo features. 
Furthermore, benchmarking with other rear-irradiance calculation software has been performed on several occasions [@Ayala2018b; @DiOrio2018; @Capelle2019]. Rear-irradiance calculation software fall into two categories: view-factor and ray-tracing models. View factor models assume isotropic scattering of reflected rays, allowing for calculation of irradiance by integration [@Marion2017]. Due-diligence software such as PVSyst or SAM use the view-factor model [@PVSyst; @SAM]. There are also some open-source view-factor models, such as bifacialvf, and PVFactors [@bifacialvf; @PVfactors]. Ray-tracing models simulate multipath reflection and absorption of individual rays entering a scene. Raytracing software such as bifacial_radiance, which is the only available open-source toolkit, offers the possibility of reproducing complex scenes, including shading or finite-system edge effects. Model agreement for view factor and bifacial_radiance software is better than 2\% (absolute) when compared with measured results. [@Ayala2018b]. 

[comment]: Plans
Plans for bifacial_radiance development include the implementation of new and existing models, addition of functionality to assist with input/output, and improvements to API consistency.

# Acknowledgements

The authors acknowledge and thank the code, documentation, and discussion contributors to the project.

S.A.P. and C.D. acknowledge support from the U.S. Department of Energy’s Solar Energy Technologies Office. This work was authored, in part, by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding was provided by the U.S. Department of Energy’s Office of Energy Efficiency and Renewable Energy (EERE) under Solar Energy Technologies Office Agreement Number 34910.

The National Renewable Energy Laboratory is a national laboratory of the U.S. Department of Energy, Office of Energy Efficiency and Renewable Energy, operated by the Alliance for Sustainable Energy, LLC.

# References