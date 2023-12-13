"""Colection of functions for monte carlo simulations.
"""

import numpy as np
import pandas as pd
from numba import njit
import pvlib 
from scipy.linalg import cholesky
from scipy import stats

# from . import spectrale
# from . import temperature

class ArugmentError(Exception):
    pass

"""corrlation class
stores modeling constants and corresonding correlation coefficient to access at runtime
"""
class Corr:
    """modeling constants"""
    mc_1 = ''
    mc_2 = ''
    correlation = 0

    def __init__(self, mc_1_string, mc_2_string, corr): # -> None: // what does this -> None do
        self.mc_1 = mc_1_string
        self.mc_2 = mc_2_string
        self.correlation = corr
    
    def getModelingConstants(self)->list:
        return [self.mc_1, self.mc_2]

# size = # correlation coeff
def _symettric_correlation_matrix(corr: list[Corr])->pd.DataFrame:

    # unpack individual modeling constants
    modeling_constants = [mc for i in corr for mc in i.getModelingConstants()]

    uniques = np.unique(modeling_constants)

    # setting up identity matrix, correct (symmetric) labels for columns and rows
    identity_matrix = np.eye(len(corr))
    identity_df = pd.DataFrame(identity_matrix, columns = uniques, index=uniques)

    # still need to walk the matrix
    # -> fill in values
    # make this standalone function if bigger function cannot handle @njit
    for i in range(len(corr)): # because we want to start on the second row
       for j in range(len(corr)): # loops columns

            # diagonal entry case -> skips to start of next row
            if identity_df.iat[i, j] == 1:
                break

            # entries under diagonal {lower triangular - I}
            else:
                # gets index and column name to check against coeff
                [x, y] = [identity_df.index[i], identity_df.index[j]] # SOMETHING WEIRD IS HAPPENING HERE

                # checks each correlation coefficients attributes to see if it matches the one we want to fill in at the given index
                for relation in corr:  
                    # skip to next correlation coefficient in list
                    if [x, y] != relation.getModelingConstants():
                        pass
                    # fills in appropriate value
                    else:
                        identity_df.iat[i, j] = relation.correlation
                    ### ADD NO CORELATION CASE ###
                    # -> fill in zero
            
    # mirror the matrix
    # // skip this for now


    # identity_df should be renamed more appropriately 
    return identity_df
