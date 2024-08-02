"""
Collection of functions to calculate diffusion of diffusants/solutes into a host material
"""

import numpy as np
import pandas as pd

def _calc_diff_substeps(
    water_new, water_old, n_steps, t, delta_t, dis, delta_dis, Fo
) -> None:  # inplace
    water_copy = water_old.copy()

    for _ in range(n_steps):
        for y in range(1, water_new.shape[0] - 1):
            # update the edge
            water_copy[0, :] = dis
            water_new[0, :] = dis + delta_dis  # one further, do we want this?

            water_new[-1, -1] = (
                Fo * (water_copy[-2, -1] * 4) + (1 - 4 * Fo) * water_copy[-1, -1]
            )  # inner node

            water_new[y, -1] = (
                Fo
                * (
                    water_copy[y + 1, -1]
                    + water_copy[y - 1, -1]
                    + 2 * water_copy[y, -2]
                )
                + (1 - 4 * Fo) * water_copy[y, -1]
            )  # internal edge

            for x in range(y + 1, water_copy.shape[1] - 1):
                neighbor_sum = (
                    water_copy[y - 1, x]
                    + water_copy[y + 1, x]
                    + water_copy[y, x - 1]
                    + water_copy[y, x + 1]
                )
                water_new[y, x] = (
                    Fo * neighbor_sum + (1 - 4 * Fo) * water_copy[y, x]
                )  # central nodes

            water_new[y, y] = (
                Fo * (2 * water_copy[y - 1, y] + 2 * water_copy[y, y + 1])
                + (1 - 4 * Fo) * water_copy[y, y]
            )  # diagonal

            dis += delta_dis
            t += delta_t  # do we need this one, temperature is not used anywhere?

            water_copy = water_new.copy()  # update copy so we can continue updating it


def module_front(
    time_index: pd.Index, # pandas index containing np.timedelta64
    backsheet_moisture: pd.Series, # g/m^3
    sample_temperature: pd.Series, # K
    p=0.1,  # [cm] perimiter area
    CW=15.6,  # [cm] cell dimensions
    nodes=20,  # number of nodes on each axis, square cell only
    eva_diffusivity_ea=0.395292897,  # eV
    Dif=2.31097881676966,  # cm^2/s diffusivity prefactor
    n_steps=20,  # number of substeps
) -> np.ndarray:
    """
    Calculate water intrusion into the front of the module 
    using 2 dimensional finite difference method.

    If this method raises any non-fatal runtime overflow errors,
    the fourier number has come above the upwards bound of 0.25.
    Increase the number of substeps, `n_steps`

    Parameters:
    -----------
    time_index: pd.Index
        pandas instance with dtype, np.timedeltad64
    backsheet_moisture: pd.Series
        water content in the backsheet of a module [g/m^3]
    sample_temperature: pd.Series
        temperature of the module [K]
    p: float
        cell perimiter area [cm]
    CW: float
        cell edge dimension, only supports square modules [cm]
    nodes: int
        number of nodes to split each axis into for finite difference method analysis. 
        higher is more accurate but slower. [unitless]
    eva_diffusivity_ea: float
        encapsulant diffusion activation energy [eV]
    Dif: float
        prefactor encapsulant diffusion [cm^2/s]
    n_steps: int
        number of stubsteps to calculate for numerical stability. 
        4-5 works for most cases but under fast changes, error can 
        accumulate quickly so 20 is a good value for numerical safety.

    Returns:
    --------
    results: np.ndarray
        3d dimensional numpy array containing a 2 dimensional numpy matrix at each timestep corresponding to water intrusion. Shape (time_index.shape[0], nodes, nodes) [g/cm^3]
    """

    EaD = eva_diffusivity_ea / 0.0000861733241  # k in [eV/K]
    W = ((CW + 2 * p) / 2) / nodes  #

    # two options, we can have a 3d array that stores all timesteps
    results = np.zeros((len(time_index) + 1, nodes, nodes))
    results[0, 0, :] = backsheet_moisture.iloc[0]

    for i in range(
        len(time_index) - 1
    ):  # loop over each entry the the results
        Temperature = sample_temperature.iloc[i]
        DTemperature = (
            sample_temperature.iloc[i + 1]
            - sample_temperature.iloc[i]
        )
        Disolved = backsheet_moisture.iloc[i]
        DDisolved = (
            backsheet_moisture.iloc[i + 1]
            - backsheet_moisture.iloc[i]
        )

        time_step = (
            (
                time_index.values[i + 1]
                - time_index.values[i]
            )
            .astype("timedelta64[s]")
            .astype(int)
        )  # timestep in units of seconds
        Fo = Dif * np.exp(-EaD / (273.15 + Temperature)) * time_step / (W * W)

        _calc_diff_substeps(
            water_new=results[i + 1, :, :],
            water_old=results[i, :, :],
            n_steps=n_steps,
            t=Temperature,
            delta_t=DTemperature,
            dis=Disolved,
            delta_dis=DDisolved,
            Fo=Fo,
        )

    return results

