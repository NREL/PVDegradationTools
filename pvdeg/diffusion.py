"""
<<<<<<< HEAD
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

=======
Collection of classes and functions to calculate diffusion of permeants into a PV module.
"""

import os
import json
import pandas as pd
from pvdeg import DATA_DIR
from numba import jit
import numpy as np
from typing import Callable

def esdiffusion(
    temperature, 
    edge_seal="OX005", 
    encapsulant="OX003", 
    edge_seal_width=1.5, 
    encapsulant_width=10, 
    seal_nodes=20, 
    encapsulant_nodes=50,
    press = 0.209,
    repeat = 1,
    Dos=None, Eads=None, Sos=None,Eass=None, Doe=None, Eade=None, Soe=None, Ease=None,
    react_func = None,
    deg_func = None,
    deg = None,
    perm = None,
    printout = True,
    **kwarg
    ):

    """
    Calculate 1-D diffusion into the edge of a PV module. This assumes an edge seal and a limited length of encapsulant. 
    In the future it will be able to run calculations for degradation and for water ingress, but initially I'm just 
    writing it to run calculations for oxygen ingress.

    Parameters
    ----------
    temperature : (pd.dataframe)
        Data Frame with minimum requirement of 'module_temperature' [°C] and 'time' [h].
    edge_seal : str, optional
        This is the name of the water or the oxygen permeation parameters for the edge seal material. 
        If left at "None" you can include the parameters (Dos, Eads, Sos, Eass) as key word arguments or use the defaults.
    encapsulant : str, optional
        This is the name of the water or the oxygen permeation parameters for the encapsulant material. 
        If left at "None" you can include the parameters (Doe, Eade, Soe, Ease) as key word arguments or use the defaults.
    edge_seal_width : float, required
        This is the width of the edge seal in [cm].
    encapsulant_width : float, required
        This is the width of the encapsulant in [cm]. 
        This assumes a center line of symmetry at the end of the encapsulant with a total module width of 2*(esw + encw)
    seal_nodes : integer, required
        This is the number of nodes used for the calculation in the edge seal.
    encapsulant_nodes : integer, required
        This is the number of nodes used for the calculation in the encapsulant.
    press : float, optional
        This is the partial pressure of oxygen.
    repeat : integer, optional
        This is the number of times to do the calculation for the whole dataset. E.g. repeat the 1-y data for 10 years.
    react_func : string, optional
        This is the name of the function that will be calculating the consumption of oxygen.
    deg_func :string, optional
        This is the name of the function that will be calculating the degradation.
    printout : Boolean
        This allows you to suppress printing messages during code execution by setting it to false.
    deg : Numpy Array
        One can send in an array with predefined degradation data already in it if desired. 
        I.e. you can have some pre degradation or areas that require more degradation.
    perm : Numpy Array
        One can send in an array with the permeant already in it if desired.
    kwargs : dict, optional
        If edge_seal or encapsulant are set at 'None' then you can enter your own parameters for, Dos, Eads, Sos, Eass, Doe, Eade, Soe, Ease in units of
        [cm²/s], [g/cm³], or [kJ/mol] for diffusivity, solubility, or activation energy respectively. If specific parameters are provided,
        then the JSON ones will be overridden.
        Should also contain any key word arguments that need to be passed to the function calculating consumption of the permeant or degradation.

    Returns
    -------
    ingress_data : pandas.DataFrame
        This will give the concentration profile as a function of time.
        If there is a degradation function called, this data will also be inclueded on a node by node basis under a third index.
    """

    with open(os.path.join(DATA_DIR, "O2permeation.json")) as user_file:
        O2 = json.load(user_file)
        user_file.close()
    with open(os.path.join(DATA_DIR, "H2Opermeation.json")) as user_file:
        H2O = json.load(user_file)
        user_file.close()

    if edge_seal[0:2]=="OX":
        esp = O2.get(edge_seal)
        if printout:
            print("Oxygen ingress parameters loaded for the edge seal.")
    else:
        if edge_seal[0:1]=="W":
            esp = H2O.get(edge_seal)
            if printout:
                print("Water ingress parameters loaded for the edge seal.")
        else:
            print("Edge seal material not found")

    if encapsulant[0:2]=="OX":
        encp = O2.get(encapsulant)
        if printout:
            print("Oxygen ingress parameters loaded for the encapsulant.")
    else:
        if encapsulant[0:1]=="W":
            encp = H2O.get(encapsulant)
            if printout:
                print("Water ingress parameters loaded for the eencapsulant.")
        else:
            print("Encapsulant material not found")
    if printout:
        try:
            print("The edge seal is", esp.get("name"), ".")
            print("The encapsulant is", encp.get("name"), ".")
        except:
            print("Unknown material selected.")

    # These are the edge seal oxygen or water permeation parameters
    if Dos == None:
        Dos = esp.get("Do")
    if Eads == None:
        Eads = esp.get("Ead") / 0.0083144626
    else:
        Eads = Eads / 0.0083144626
    if Sos == None:
        Sos = (
            esp.get("So") * press
        )  # puts in the adjustment for the atmospheric pressure or oxygen.
    if Eass == None:
        Eass = esp.get("Eas") / 0.0083144626
    else:
        Eass = Eass / 0.0083144626
    # These are the encapsulant oxygen permeaiton parameters
    if Doe == None:
        Doe = encp.get("Do")
    if Eade == None:
        Eade = encp.get("Ead") / 0.0083144626
    else:
        Eade = Eade / 0.0083144626
    if Soe == None:
        Soe = (
            encp.get("So") * press
        )  # puts in the adjustment for the atmospheric pressure or oxygen.
    if Ease == None:
        Ease = encp.get("Eas") / 0.0083144626
    else:
        Ease = Ease / 0.0083144626

    so = Sos / Soe
    eas = Eass - Ease
    dod = Dos / Doe
    ead = Eads - Eade

    edge_seal_width = (
        edge_seal_width / (seal_nodes - 0.5)
    )  # The 0.5 is put in there because the model exterior is defined as the center of the edge node.
    encapsulant_width = (
        encapsulant_width / (encapsulant_nodes - 0.5)
    )  # The 0.5 is put in because the model interior encapsulant node is a point of symmetry and defines the condition at the center line.

    perm_mid = np.array(
        np.zeros((seal_nodes + encapsulant_nodes + 3)), dtype=np.float64
        )  # This is the profile at a transition point between output points.
    if perm == None:
        perm = np.array(
            np.zeros(
                (len(temperature) * repeat - repeat + 1, seal_nodes + encapsulant_nodes + 3), dtype=np.float64
            )
        )  # It adds in two nodes for the interface concentration for both materials and one for the hour column.

    temperature = pd.DataFrame(
        temperature, columns=["module_temperature", "time", "time_step"]
    )  # This adds the number of time steps to be used as a subdivision between data points. [s]
    met_data = temperature[["module_temperature", "time"]].to_numpy(dtype=np.float32)
    met_data[:, 0] = met_data[:, 0] + 273.15
    time_step = np.array(np.ones(len(temperature)), dtype=np.int8)

    for row in range(
        0, len(met_data) - 1
    ):  # This section sets up the number of sub calculations for each output point to ensure model calculation stability.
        dt = (met_data[row + 1][1] - met_data[row][1]) * 3600
        fos = Dos * np.exp(-Eads / met_data[row][0]) * dt / edge_seal_width / edge_seal_width
        foe = (
            Doe * np.exp(-Eads / met_data[row][0]) * dt / encapsulant_width / encapsulant_width
        )  # This is the dimensionless Fourrier number for the encapsulant. Diffusivity [=] cm2/s, T [=] s, w [=] cm
        f_max = 0.25
        if fos > f_max or foe > f_max:
            if fos > foe:
                time_step[row] = np.trunc(fos / f_max) + 1
            else:
                time_step[row] = np.trunc(foe / f_max) + 1
    if deg_func != None and deg == None: # Sets up an array to do the degradation calculation.
        deg=perm
    perm[0][1] = Sos * np.exp(-Eass / met_data[0][0])
    perm_mid = perm[0]
    for rp_num in range(repeat):
        rp_time = rp_num * met_data[met_data.shape[0] - 1][1]
        # -1 is used because it doesn't know how much time to have from the end of the
        # data to a repeat of the data, it just ignores the loop step.
        rp_row = rp_num * (met_data.shape[0] - 1)
        for row in range(0, len(temperature) - 1):
            # Time step in [s]
            dt = (met_data[row + 1][1] - met_data[row][1]) * 3600 / time_step[row + 1]
            # Temperature step in [K]
            dtemp = (met_data[row + 1][0] - met_data[row][0]) / time_step[row + 1]
            for mid_point in range(1, time_step[row + 1] + 1):
                fos = (
                    Dos * np.exp(-Eads / (met_data[row][0] + dtemp * mid_point))
                    * dt / edge_seal_width / edge_seal_width
                )
                foe = (
                    Doe * np.exp(-Eade / (met_data[row][0] + dtemp * mid_point))
                    * dt / encapsulant_width / encapsulant_width
                )
                # Cs edge seal/Ce encapsulant
                r1 = so * np.exp(-eas / (met_data[row][0] + dtemp * mid_point))
                r2 = dod * np.exp(-ead / (met_data[row][0] + dtemp * mid_point)
                                  )* r1 * encapsulant_width / edge_seal_width
                  # Ds/De*Cs/Ce*We/Ws
                # Calculates the edge seal nodes. Adjusted to not calculate ends and to have the first node be temperature.
                for node in range(2, seal_nodes):
                    perm[row + 1 + rp_row][node] = perm_mid[node] + fos * (
                        perm_mid[node - 1] + perm_mid[node + 1] - 2 * perm_mid[node]
                    )
                # calculates the encapsulant nodes. Adjust to not calculated ends and for a prior temperature and two interface nodes.
                for node in range(seal_nodes + 4, encapsulant_nodes + seal_nodes + 2):
                    perm[row + 1 + rp_row][node] = perm_mid[node] + foe * (
                        perm_mid[node - 1] + perm_mid[node + 1] - 2 * perm_mid[node]
                    )
                # Calculates the center encapsulant node. Accounts for temperature and two interface nodes.
                perm[row + 1 + rp_row][encapsulant_nodes + seal_nodes + 2] = perm_mid[
                    encapsulant_nodes + seal_nodes + 2] + 2 * foe * (
                        perm_mid[encapsulant_nodes + seal_nodes + 1] - 
                        perm_mid[encapsulant_nodes + seal_nodes + 2])
                
                # Calculated edge seal node adjacent to the first encapsulant node. Node numbers shifted.
                perm[row + 1 + rp_row][seal_nodes] = perm_mid[seal_nodes] + fos * (
                    perm_mid[seal_nodes - 1]
                    + perm_mid[seal_nodes + 3] * r1 * 2 / (1 + r2)
                    - perm_mid[seal_nodes] * (1 + 2 / (1 + r2)))
  
                # Calculated encapsulant node adjacent to the last edge seal node. Node numbers shifted.
                perm[row + 1 + rp_row][seal_nodes + 3] = perm_mid[seal_nodes + 3] + foe * (
                    perm_mid[seal_nodes] / r1 * 2 / (1 + 1 / r2)
                    + perm_mid[seal_nodes + 4]
                    - perm_mid[seal_nodes + 3] * (1 + 2 / (1 + 1 / r2)))
                
                # sets the concentration at the edge seal to air interface.
                perm[row + 1 + rp_row][1] = Sos * np.exp(
                    -Eass / (met_data[row + 1][0] + dtemp * mid_point)
                )


                # Runs the degradation calculation.            
                if deg_func != None:
                    print('oops')
                # Runs the reaction with permeant function.
                if react_func != None:
                    print('oops')
                    
                perm_mid = perm[row + 1 + rp_row]

            # Calculate edge seal at interface to encapsulant.
            # Blocked out code did weird things and was based on equal fluxes. Actually using a simple averaging. This looks better and is not used in the diffusion calculations.
            #perm[row + 1 + rp_row][seal_nodes + 1] = (perm_mid[seal_nodes + 3]*r1  
            #                                          + perm_mid[seal_nodes]*r2) / (1+r2)
            perm[row + 1 + rp_row][seal_nodes + 1] = perm_mid[seal_nodes ]+(perm_mid[seal_nodes]-perm_mid[seal_nodes-1])/2

            # Calculate encapsulant at interface to the edge seal.
            #perm[row + 1 + rp_row][seal_nodes + 2] = perm[row + 1 + rp_row][seal_nodes + 1] / r1
            perm[row + 1 + rp_row][seal_nodes + 2] = perm_mid[seal_nodes + 3]+(perm_mid[seal_nodes + 4]-perm_mid[seal_nodes+3])/2

            # Puts in the time for the first column.
            perm[row + 1 + rp_row][0] = rp_time + met_data[row + 1][1]



        # Because it is cycling around, it needs to start with the last temperature.
        met_data[0][0] = met_data[met_data.shape[0] - 1][0]

    positions = np.array(np.zeros((seal_nodes + encapsulant_nodes + 3)))
    for node in range(seal_nodes):  # Creates the header values of the distance.
        positions[node + 1] = node * edge_seal_width
    positions[seal_nodes + 1] = seal_nodes * edge_seal_width - edge_seal_width / 2
    positions[seal_nodes + 2] = positions[seal_nodes + 1]
    for node in range(encapsulant_nodes):
        positions[seal_nodes + 3 + node] = positions[seal_nodes + 1] + encapsulant_width / 2 + encapsulant_width * node
    perm = np.vstack([positions, perm])

    return pd.DataFrame(perm[1:, 1:], index=perm[1:, 0], columns=perm[0, 1:])
>>>>>>> development
