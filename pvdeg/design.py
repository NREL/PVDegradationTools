"""Collection of functions for PV module design considertations.
"""

from . import humidity


def edge_seal_ingress_rate(avg_psat):
    """
    This function generates a constant k, relating the average moisture ingress rate through a
    specific edge seal, Helioseal 101. Is an emperical estimation the rate of water ingress of
    water through edge seal material. This function was determined from numerical calculations
    from several locations and thus produces typical responses. This simplification works
    because the environmental temperature is not as important as local water vapor pressure.
    For the same environmental water concentration, a higher temperature results in lower
    absorption in the edge seal but lower diffusivity through the edge seal. In practice, these
    effects nearly cancel out makeing absolute humidity the primary parameter determining
    moisture ingress through edge seals.

    See: Kempe, Nobles, Postak Calderon,"Moisture ingress prediction in polyisobutylene‐based
    edge seal with molecular sieve desiccant", Progress in Photovoltaics, DOI: 10.1002/pip.2947

    Parameters
    -----------
    avg_psat : float
        Time averaged time averaged saturation point for an environment in kPa.
        When looking at outdoor data, one should average over 1 year

    Returns
    -------
    k : float [cm/h^0.5]
        Ingress rate of water through edge seal.
        Specifically it is the ratio of the breakthrough distance X/t^0.5.
        With this constant, one can determine an approximate estimate of the ingress distance
        for a particular climate without more complicated numerical methods and detailed
        environmental analysis.

    """

    k = 0.0013 * (avg_psat) ** 0.4933

    return k


def edge_seal_width(weather_df, meta, k=None, years=25, from_dew_point=False):
    """
    Determine the width of edge seal required for given number of years water ingress.

    Parameters
    ----------
    weather_df : pd.DataFrame
        must be datetime indexed and contain at least temp_air, temp_dew
    meta : dict
        location meta-data (from weather file)
    k: float
        Ingress rate of water through edge seal. [cm/h^0.5]
        Specifically it is the ratio of the breakthrough distance X/t^0.5.
        See the function design.edge_seal_ingress_rate()
    years : integer, default = 25
        Integer number of years under water ingress
    from_dew_point : boolean, optional
        If true, will compute the edge seal width from temp_dew instead of dry bulb air temp

    Returns
    ----------
    width : float
        Width of edge seal required for input number of years water ingress. [cm]
    """

    if from_dew_point:
        # "Dew Point" fallback handles key-name bug in pvlib < v0.10.3.
        temp = weather_df.get("temp_dew", weather_df.get("Dew Point"))
    else:
        temp = weather_df["temp_air"]

    if k is None:
        psat, avg_psat = humidity.psat(temp)
        k = edge_seal_ingress_rate(avg_psat)

    width = k * (years * 365.25 * 24) ** 0.5

    return width


# TODO: Include gaps functionality
