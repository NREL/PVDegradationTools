"""Collection of functions for PV module design considertations.
"""

import numpy as np
import pandas as pd
from numba import jit
from rex import NSRDBX
from rex import Outputs
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import humidity


def edge_seal_width(k, years):
    """
    Determine the width of edge seal required for given number of years water ingress.

    Parameters
    ----------
    k: float
        Ingress rate of water through edge seal. [cm/h^0.5]
        Specifically it is the ratio of the breakthrough distance X/t^0.5.

    Returns
    ----------
    width : float 
        Width of edge seal required for a 25 year water ingress. [cm]
    """

    width = k * (years * 365.25 * 24)**.5

    return width

#TODO: Where is dew_pt_temp coming from?
def edge_seal_from_dew_pt(dew_pt_temp, years):
    """
    Compute the edge seal width required for 25 year water ingress directly from
    dew pt tempterature.

    Parameters
    ----------
    dew_pt_temp : float, or float series
        Dew Point Temperature
    all_results : boolean
        If true, returns all calculation steps: psat, avg_psat, k, edge seal width
        If false, returns only edge seal width

    Returns
    ----------
    edge_seal_width: float
        Width of edge seal [mm] required for 25 year water ingress

    Optional Returns
    ----------
    psat : series
        Hourly saturation point
    avg_psat : float
        Average saturation point over sample times
    k : float
        Ingress rate of water vapor
    """
    
    psat = humidity.psat(dew_pt_temp)
    avg_psat = psat.mean()

    k = .0013 * (avg_psat)**.4933

    edge_seal_width = edge_seal_width(k, years)

    res = {'psat':psat,
           'avg_psat':avg_psat,
           'k':k,
           'edge_seal_width':edge_seal_width}

    return res


#TODO: Include gaps functionality