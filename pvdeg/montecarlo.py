"""Colection of functions for monte carlo simulations.
"""

### TODO:

# Seperate calculate function?

# implement modular arrhenius implementation (used in jupyter notebook)
# calculate : pd.DataFrame output
# first do old case to make sure it works, add robustness testing
# we dont have a function for the previous test case
# then do standoff calculation (pvdeg.standards.standoff())

import numpy as np
import pandas as pd
from numba import njit
from scipy.linalg import cholesky
from scipy import stats


class Corr:     # could be made into a dataclass
    """corrlation class
    stores modeling constants and corresponding correlation coefficient to access at runtime
    """

    # modeling constants : str
    mc_1 = ''
    mc_2 = ''
    # corresonding corelation coefficient : float
    correlation = 0

    def __init__(self, mc_1_string, mc_2_string, corr): 
        """parameterized constructor"""
        self.mc_1 = mc_1_string
        self.mc_2 = mc_2_string
        self.correlation = corr
    
    def getModelingConstants(self)->list[str, str]:
        """
        Helper method. Returns modeling constants in string form.

        Parameters
        ----------
        self : Corr
            Reference to self

        Returns
        ----------
        modeling_constants : list[str, str]
            Both modeling constants in string from from their corresponding correlation coefficient object
        """

        modeling_constants = [self.mc_1, self.mc_2]
        return modeling_constants

def _symettric_correlation_matrix(corr: list[Corr])->pd.DataFrame:
    """
    Helper function. Generate a symmetric correlation coefficient matrix.

    Parameters
    ----------
    corr : list[Corr]
        All correlations between appropriate modeling constants

    Returns
    ----------
    identity_df : pd.DataFrame
        Matrix style DataFrame containing relationships between all input modeling constants
        Index and Column names represent modeling constants for comprehensibility
    """

    # unpack individual modeling constants from correlations
    modeling_constants = [mc for i in corr for mc in i.getModelingConstants()]

    uniques = np.unique(modeling_constants)

    # setting up identity matrix, labels for columns and rows
    identity_matrix = np.eye(len(uniques)) 
    identity_df = pd.DataFrame(identity_matrix, columns = uniques, index=uniques)

    # walks matrix to fill in correlation coefficients
    # make this a modular standalone function if bigger function preformance is not improved with @njit 
    for i in range(len(uniques)):
        for j in range(i):  # only iterate over lower triangle
            x, y = identity_df.index[i], identity_df.columns[j]

            # find the correlation coefficient
            found = False
            for relation in corr:
                if set([x, y]) == set(relation.getModelingConstants()):
                    # fill in correlation coefficient
                    identity_df.iat[i, j] = relation.correlation
                    found = True
                    break

            # if no matches in all correlation coefficients, they will be uncorrelated (= 0)
            if not found:
                identity_df.iat[i, j] = 0  

    # mirror the matrix
    # this may be computationally expensive for large matricies
    # could be better to fill the original matrix in all in one go rather than doing lower triangular and mirroring it across I
    identity_df = identity_df + identity_df.T - np.diag(identity_df.to_numpy().diagonal())

    # identity_df should be renamed more appropriately 
    return identity_df

def _createStats(stats : dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    helper function. Unpacks mean and standard deviation for modeling constants into a DataFrame

    Parameters
    ----------
    stats : dict[str, dict[str, float]]
        contains mean and standard deviation for each modeling constant
        example of one mc:  {'Ea' : {'mean' : 62.08, 'stdev' : 7.3858 }}
    
    Returns
    ----------
    stats_df : pd.DataFrame
        contains unpacked means and standard deviations from dictionary
    """

    # incomplete dataset
    for mc in stats:
        if 'mean' not in stats[mc] or 'stdev' not in stats[mc]:
            raise ValueError(f"Missing 'mean' or 'stdev' for modeling constant")

    # unpack data 
    modeling_constants = list(stats.keys())
    mc_mean = [stats[mc]['mean'] for mc in modeling_constants]
    mc_stdev = [stats[mc]['stdev'] for mc in modeling_constants]
    idx = ['mean', 'stdev']

    stats_df = pd.DataFrame({'mean' : mc_mean, 'stdev' : mc_stdev}, index=modeling_constants).T

    return stats_df

def _correlateData(samples_to_correlate : pd.DataFrame, stats_for_correlation : pd.DataFrame) -> pd.DataFrame:

    """
    helper function. Uses meaningless correlated samples and makes meaningful by 
    multiplying random samples by their parent modeling constant's standard deviation
    and adding the mean

    Parameters
    ----------
    samples_to_correlate : pd.DataFrame
        contains n samples generated with N(0, 1) for each modeling constant
        column names must be consistent with all modeling constant inputs

    stats_for_correlation : pd.DataFrame
        contains mean and stdev each modeling constant,
        column names must be consistent with all modeling constant inputs

    Returns
    ----------
    correlated_samples : pd.DataFrame
        correlated samples in a tall dataframe. column names match modeling constant inputs,
        integer indexes. See generateCorrelatedSamples() references section for process info
    """

    # accounts for out of order column names, AS LONG AS ALL MATCH
    # UNKNOWN CASE: what will happen if there is an extra NON matching column in stats
    columns = list(samples_to_correlate.columns.values)
    ordered_stats = stats_for_correlation[columns]

    # SHOULD CHANGE FROM ILOC TO "mean" and "stdev" FOR ADAPTABILITY
    correlated_samples = samples_to_correlate.multiply(ordered_stats.iloc[1]).add(ordered_stats.iloc[0])

    return correlated_samples

def generateCorrelatedSamples(corr : list[Corr], stats : dict[str, dict[str, float]], n : int) -> pd.DataFrame:
    # columns are now named, may run into issues if more mean and stdev entries than correlation coefficients
    # havent tested yet but this could cause major issues (see lines 163 and 164 for info)

    """
    Generates a tall correlated samples numpy array based on correlation coefficients and mean and stdev 
    for modeling constants. Values are correlated from cholesky decomposition of correlation coefficients,
    and n random samples for each modeling constant generated from a standard distribution with mean = 0
    and standard deviation = 1.

    Parameters
    ----------
    corr : List[Corr]

    stats : 

    n : int
        number of samples to create

    Returns
    ----------
    correlated_samples : pd.Dataframe
        tall dataframe of dimensions (n by # of modeling constants).
        Columns named as modeling constants from Corr object inputs

    References
    ----------
    Burgess, Nicholas, Correlated Monte Carlo Simulation using Cholesky Decomposition (March 25, 2022). 
    Available at SSRN: https://ssrn.com/abstract=4066115 
    """

    coeff_matrix = _symettric_correlation_matrix(corr)

    decomp = cholesky(coeff_matrix.to_numpy(), lower = True)

    samples = np.random.normal(loc=0, scale=1, size=(len(stats), n)) 
    
    precorrelated_samples = np.matmul(decomp, samples) 

    precorrelated_df = pd.DataFrame(precorrelated_samples.T, columns=coeff_matrix.columns.to_list())

    stats_df = _createStats(stats)    

    correlated_df = _correlateData(precorrelated_df, stats_df)

    return correlated_df

# this shouldn't stay here but I thought it was best for short term cleanlyness sake
def temp_arrhenius():
    # port from jupyter notebook

    return

def simulate():
    # similar to pvdeg.geospatial.analyis 

    return


# monte carlo function
# model after - https://github.com/NREL/PVDegradationTools/blob/main/pvdeg_tutorials/tutorials/LETID%20-%20Outdoor%20Geospatial%20Demo.ipynb
# Define desired analysis
# geo = {'func': pvdeg.letid.calc_letid_outdoors,
#        'weather_ds': weather_SW_sub,
#        'meta_df': meta_SW_sub,
#        'tau_0': 115, # us, carrier lifetime in non-degraded states, e.g. LETID/LID states A or C
#        'tau_deg': 55, # us, carrier lifetime in fully-degraded state, e.g. LETID/LID state B
#        'wafer_thickness': 180, # um
#        's_rear': 46, # cm/s
#        'cell_area': 243, # cm^2
#        'na_0': 100,
#        'nb_0': 0,
#        'nc_0': 0,
#        'mechanism_params': 'repins'
# }

# letid_res = pvdeg.geospatial.analysis(**geo)
# -----------------------
# geospatial.analysis()


# to call function we want to collect parameters in a modular way
# --------------------------------------- #

"""
# parameters for simulation
# function and input data
sim =   {'func' : funcName(), # pvdeg.montecarlo.temp_arrhenius 
         'weather_df' : pd.DataFrame, # could be datastructure
         'meta_df' : pd.DataFrame 
}

# paramters to CONTROL simulation
monte = {'trials' : int,
         'correlations' : list[Corr],
         'stats' : pd.DataFrame
}
"""