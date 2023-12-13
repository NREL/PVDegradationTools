"""Colection of functions for monte carlo simulations.
"""

### TODO:
# generate correlated samples
# implement existing pvdeg calculation functions
# calculate using ↑
# output in suitable format (dataframe?)

import numpy as np
import pandas as pd
from numba import njit
from scipy.linalg import cholesky
from scipy import stats

"""corrlation class
stores modeling constants and corresponding correlation coefficient to access at runtime
"""
class Corr:     # could be made into a dataclass
    """modeling constants"""
    mc_1 = ''
    mc_2 = ''
    """corresponding correlation coefficient"""
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

# statistical data capture,
# mean and stdev
# for simplicity sake
# this should happen before creating the correlation matrix

# we already have list of correlation coefficients, 
# unpack them
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

    for mc in stats:
        if 'mean' not in stats[mc] or 'stdev' not in stats[mc]:
            raise ValueError(f"Missing 'mean' or 'stdev' for modeling constant")

    modeling_constants = list(stats.keys())
    mc_mean = [stats[mc]['mean'] for mc in modeling_constants]
    mc_stdev = [stats[mc]['stdev'] for mc in modeling_constants]
    
    idx = ['mean', 'stdev']

    stats_df = pd.DataFrame({'mean' : mc_mean, 'stdev' : mc_stdev}, index=modeling_constants)

    return stats_df