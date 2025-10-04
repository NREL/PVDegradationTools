"""Collection of functions for monte carlo simulations."""

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from typing import Callable
import inspect


class Corr:
    """
    Correlation class.

    corrlation class :
    stores modeling constants and corresponding correlation coefficient to access at
    runtime
    """

    # modeling constants : str
    mc_1 = ""
    mc_2 = ""
    # corresonding corelation coefficient : float
    correlation = 0

    def __init__(self, mc_1_string, mc_2_string, corr):
        """Parameterized constructor."""
        self.mc_1 = mc_1_string
        self.mc_2 = mc_2_string
        self.correlation = corr

    def getModelingConstants(self) -> list[str, str]:
        """Return modeling constants in string form, helper method.

        Parameters
        ----------
        self : Corr
            Reference to self

        Returns
        ----------
        modeling_constants : list[str, str]
            Both modeling constants in string from from their corresponding correlation
            coefficient object
        """
        modeling_constants = [self.mc_1, self.mc_2]
        return modeling_constants


def _symettric_correlation_matrix(corr: list[Corr]) -> pd.DataFrame:
    """Generate a symmetric correlation coefficient matrix, helper function.

    Parameters
    ----------
    corr : list[Corr]
        All correlations between appropriate modeling constants

    Returns
    ----------
    identity_df : pd.DataFrame
        Matrix style DataFrame containing relationships between all input modeling
        constants. Index and Column names represent modeling constants for
        comprehensibility.
    """
    if not corr:
        return None

    # unpack individual modeling constants from correlations
    modeling_constants = [mc for i in corr for mc in i.getModelingConstants()]

    uniques = np.unique(modeling_constants)

    # setting up identity matrix, labels for columns and rows
    identity_matrix = np.eye(len(uniques))
    identity_df = pd.DataFrame(identity_matrix, columns=uniques, index=uniques)

    # walks matrix to fill in correlation coefficients
    # make this modular standalone function if performance is not improved with @njit
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

            # if no matches in all correlation coefficients, they will be uncorrelated
            # (= 0)
            if not found:
                identity_df.iat[i, j] = 0

    # mirror the matrix
    # this may be computationally expensive for large matricies
    # could be better to fill the original matrix in all in one go rather than doing
    # lower triangular and mirroring it across I
    identity_df = (
        identity_df + identity_df.T - np.diag(identity_df.to_numpy().diagonal())
    )

    # identity_df should be renamed more appropriately
    return identity_df


def _createStats(stats: dict[str, dict[str, float]], corr: list[Corr]) -> pd.DataFrame:
    """Unpack mean and standard deviation for modeling constants into a DataFrame.

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
    # empty correlation list case
    if not corr:
        stats_df = pd.DataFrame(stats)
        return stats_df

    # incomplete dataset
    for mc in stats:
        if "mean" not in stats[mc] or "stdev" not in stats[mc]:
            raise ValueError("Missing 'mean' or 'stdev' for modeling constant")

    # unpack data
    modeling_constants = list(stats.keys())
    mc_mean = [stats[mc]["mean"] for mc in modeling_constants]
    mc_stdev = [stats[mc]["stdev"] for mc in modeling_constants]

    stats_df = pd.DataFrame(
        {"mean": mc_mean, "stdev": mc_stdev}, index=modeling_constants
    ).T

    # flatten and reorder
    modeling_constants = [mc for i in corr for mc in i.getModelingConstants()]
    uniques = np.unique(modeling_constants)

    # what happens if columns do not match?
    if len(uniques) != len(corr):
        raise ValueError("correlation data is insufficient")

    # should match columns from correlation matrix
    stats_df = stats_df[uniques]

    return stats_df


def _correlateData(
    samples_to_correlate: pd.DataFrame, stats_for_correlation: pd.DataFrame
) -> pd.DataFrame:
    """Manipulate correlated samples, helper function.

    Uses meaningless correlated samples and makes meaningful by
    multiplying random samples by their parent modeling constant's standard deviation
    and adding the mean.

    Parameters
    ----------
    samples_to_correlate : pd.DataFrame
        contains n samples generated with N(0, 1) for each modeling constant
        column names must be consistent with all modeling constant inputs
    stats_for_correlation : pd.DataFrame
        contains mean and stdev each modeling constant,
        column names must be consistent with all modeling constant inputs

    Returns
    -------
    correlated_samples : pd.DataFrame
        correlated samples in a tall dataframe. column names match modeling constant
        inputs, integer indexes. See generateCorrelatedSamples() references section for
        process info.
    """
    # accounts for out of order column names, AS LONG AS ALL MATCH
    # UNKNOWN CASE: what will happen if there is an extra NON matching column in stats
    columns = list(samples_to_correlate.columns.values)
    ordered_stats = stats_for_correlation[columns]

    means = ordered_stats.loc["mean"]
    stdevs = ordered_stats.loc["stdev"]

    correlated_samples = samples_to_correlate.multiply(stdevs).add(means)

    return correlated_samples


def generateCorrelatedSamples(
    corr: list[Corr], stats: dict[str, dict[str, float]], n: int, seed=None
) -> pd.DataFrame:
    # columns are now named, may run into issues if more mean and stdev entries than
    # correlation coefficients
    # havent tested, could cause major issues (see lines 163 and 164 for info)
    """Generate tall correlated samples np.array.

    Calculated based on correlation coefficients and mean and stdev for modeling
    constants. Values are correlated from cholesky decomposition of correlation
    coefficients, and n random samples for each modeling constant generated from a
    standard distribution with mean = 0 and standard deviation = 1.

    Parameters
    ----------
    corr : List[Corr]
        list containing correlations between variable
    stats : dict[str, dict[str, float]]
        dictionary storing variable mean and standard deviation.
        Syntax : `<variable_name> : {'mean' : <float>, 'stdev' : <float>}`
    n : int
        number of samples to create
    seed : Any, optional
        reseed the numpy BitGenerator, numpy legacy function (use cautiously)

    Returns
    -------
    correlated_samples : pd.Dataframe
        tall dataframe of dimensions (n by # of modeling constants).
        Columns named as modeling constants from Corr object inputs

    References
    ----------
    Burgess, Nicholas, Correlated Monte Carlo Simulation using Cholesky Decomposition
    (March 25, 2022). Available at SSRN: https://ssrn.com/abstract=4066115
    """
    if seed:
        np.random.seed(seed=seed)

    # base case
    if corr:
        coeff_matrix = _symettric_correlation_matrix(corr)  # moved inside

        decomp = cholesky(coeff_matrix.to_numpy(), lower=True)

        # list of correlations
        # using to check if all r = 0
        values = []
        for i in corr:
            values.append(i.correlation)

        # check if all zero
        all_zeros = all(value == 0 for value in values)

    samples = np.random.normal(loc=0, scale=1, size=(len(stats), n))

    stats_df = _createStats(stats, corr)

    # no correlation data given, only stats
    # OR, all correlations are 0
    if (not corr) or (all_zeros):
        nocorr_df = pd.DataFrame(samples.T, columns=stats_df.columns.tolist())

        meaningful_nocorr_df = _correlateData(nocorr_df, stats_df)

        return meaningful_nocorr_df

    if corr:
        precorrelated_samples = np.matmul(decomp, samples)

        precorrelated_df = pd.DataFrame(
            precorrelated_samples.T, columns=coeff_matrix.columns.to_list()
        )

        correlated_df = _correlateData(precorrelated_df, stats_df)

        return correlated_df


# monte carlo function
# model after -
# https://github.com/NREL/PVDegradationTools/blob/main/pvdeg_tutorials/tutorials/LETID%20-%20Outdoor%20Geospatial%20Demo.ipynb  # noqa


def simulate(
    func: Callable, correlated_samples: pd.DataFrame, **function_kwargs
) -> pd.Series:
    """Apply a target function to data to preform a monte carlo simulation.

    If you get
    a key error and the target function has default parameters, try adding them to your
    ``func_kwargs`` dictionary instead of using the default value from the target
    function.

    Parameters
    ----------
    func : function
        Function to apply for monte carlo simulation
    correlated_samples : pd.DataFrame
        Dataframe of correlated samples with named columns for each appropriate modeling
        constant, can be generated using generateCorrelatedSamples()
    function_kwargs : dict
        Keyword arguments to pass to func, only include arguments not named in your
        correlated_samples columns

    Returns
    -------
    res : pandas.Series
        Series with monte carlo results from target function
    """
    # NOTES
    # func modeling constant parameters must be lowercase in function definition
    # dynamically construct argument list for func
    # call func with .apply(lambda)
    args = {k.lower(): v for k, v in function_kwargs.items()}  # make lowercase

    func_signature = inspect.signature(func)

    func_args = set(func_signature.parameters.keys())

    def prepare_args(row):
        return {
            arg: row[arg] if arg in row else function_kwargs.get(arg)
            for arg in func_args
        }

    args = prepare_args(correlated_samples.iloc[0])

    def apply_func(row):
        row_args = {**args, **{k.lower(): v for k, v in row.items()}}

        return func(**row_args)

    # this line is often flagged when target function is not given required arguments
    # problems also arise when target function parameter names are not lowercase
    result = correlated_samples.apply(apply_func, axis=1)

    return result
