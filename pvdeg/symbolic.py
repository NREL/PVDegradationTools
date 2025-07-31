"""Functions to enable arbitrary symbolic expression evaluation for simple models."""

import sympy as sp
import pandas as pd
import numpy as np

# from latex2sympy2 import latex2sympy # this potentially useful but if someone has
# to use this then they proboably wont be able to figure out the rest
# parse: latex -> sympy using latex2sympy2 if nessesscary


def calc_kwarg_floats(
    expr: sp.core.mul.Mul,
    kwarg: dict,
) -> float:
    """Calculate a symbolic sympy expression using a dictionary of values.

    Parameters:
    ----------
    expr: sp.core.mul.Mul
        symbolic sympy expression to calculate values on.
    kwarg: dict
        dictionary of kwarg values for the function, keys must match
        sympy symbols.

    Returns:
    --------
    res: float
        calculated value from symbolic equation
    """
    res = expr.subs(kwarg).evalf()
    return res


def calc_df_symbolic(
    expr: sp.core.mul.Mul,
    df: pd.DataFrame,
) -> pd.Series:
    """Calculate the expression over the entire dataframe.

    Parameters:
    ----------
    expr: sp.core.mul.Mul
        symbolic sympy expression to calculate values on.
    df: pd.DataFrame
        pandas dataframe containing column names matching the sympy symbols.
    """
    variables = set(map(str, list(expr.free_symbols)))
    if not variables.issubset(df.columns.values):
        raise ValueError(
            f"""
                                 all expression variables need to be in dataframe cols
                                 expr symbols   : {expr.free_symbols}")
                                 dataframe cols : {df.columns.values}
                                 """
        )

    res = df.apply(lambda row: calc_kwarg_floats(expr, row.to_dict()), axis=1)
    return res


def _have_same_indices(series_list):
    if not series_list:
        return True

    if not isinstance(series_list, pd.Series):
        return False

    first_index = series_list[0].index

    same_indicies = all(s.index.equals(first_index) for s in series_list[1:])
    all_series = all(isinstance(value, pd.Series) for value in series_list)

    return same_indicies and all_series


def _have_same_length(series_list):
    if not series_list:
        return True

    first_length = series_list[0].shape[0]
    return all(s.shape[0] == first_length for s in series_list[1:])


def calc_kwarg_timeseries(
    expr,
    kwarg,
):
    # check for equal length among timeseries. no nesting loops allowed, no functions
    # can be dependent on their previous results values
    numerics, timeseries, series_length = {}, {}, 0
    for key, val in kwarg.items():
        if isinstance(val, (pd.Series, np.ndarray)):
            timeseries[key] = val
            series_length = len(val)
        elif isinstance(val, (int, float)):
            numerics[key] = val
        else:
            raise ValueError("only simple numerics or timeseries allowed")

    if not _have_same_length(list(timeseries.values())):
        raise NotImplementedError(
            "arrays/series are different lengths. fix mismatched length. "
            "otherwise arbitrary symbolic solution is too complex for solver. "
            "nested loops or loops dependent on previous results not supported."
        )

    # calculate the expression. we will seperately calculate all values and store then
    # in a timeseries of the same shape. if a user wants to sum the values then they can
    if _have_same_indices(list(timeseries.values())):
        index = list(timeseries.values())[0].index
    else:
        index = pd.RangeIndex(start=0, stop=series_length)
    res = pd.Series(index=index, dtype=float)

    for i in range(series_length):
        # calculate at each point and save value
        iter_dict = {
            key: value.values[i] for key, value in timeseries.items()
        }  # pandas indexing will break like this in future versions, we could only

        iter_dict = {**numerics, **iter_dict}

        # we are still getting timeseries at this point
        res.iloc[i] = float(expr.subs(iter_dict).evalf())

    return res
