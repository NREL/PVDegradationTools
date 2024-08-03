"""
Collection of classes and functions to calculate diffusion of permeants into a PV module.
"""


import json

def esdiffusion (
        temperature, 
        es=None, 
        enc=None, 
        esw=1.5, 
        encw=10, 
        sn=20, 
        en=50,
        ** kwarg
        ):
    """
    Calculate 1-D diffusion into the edge of a PV module. This assumes an edge seal and a limited length of encapsulant. 
    In the future it will be able to run calculations for degradation and for water ingress, but initially I'm just 
    writing it to run calculations for oxygen ingress.

    Parameters
    ----------
    temperature : (pd.dataframe)
        Data Frame with minimum requirement of 'module_temperature' and 'time'.
    es : str, optional
        This is the name of the water or the oxygen permeation parameters for the edge seal material. 
        If left at "None" you must include the parameters as key word arguments.
    enc : str, optional
        This is the name of the water or the oxygen permeation parameters for the encapsulant material. 
        If left at "None" you must include the parameters as key word arguments.
    esw : float, required
        This is the width of the edge seal in [cm].
    encw : float, required
        This is the width of the encapsulant in [cm]. 
        This assumes a center line of symmetry at the end of the encapsulant with a total module width of 2*(esw + encw)
    sn : integer, required
        This is the number of nodes used for the calculation in the edge seal.
    en : integer, required
        This is the number of nodes used for the calculation in the encapsulant.
    kwargs : dict, optional
        If es or enc are left at 'None' then the parameters, Dos, Eads, Sos,

    Returns
    -------
    ingress_data : pandas.DataFrame
        This will give the concentration profile as a function of temperature along with degradation parameters in futur iterations..
    """


    with open(os.path.join(DATA_DIR, 'O2permeation.json')) as user_file:
    O2= json.load(user_file)
    user_file.close()
    #O2

    es = O2.get('OX005')    #This is the number for the edge seal in the json file
    enc = O2.get('OX003')   #This is the number for the encapsulant in the json file

    #These are the edge seal oxygen permeation parameters
    Dos=es.get('Do')
    Eads=es.get('Ead')
    Sos=es.get('So')
    Eass=es.get('Eas')
    #These are the encapsulant oxygen permeaiton parameters
    Doe=enc.get('Do')
    Eade=enc.get('Ead')
    Soe=enc.get('So')
    Ease=enc.get('Eas')

    Esw = 1.5   #This is the edge seal width in [cm]
    Encw = 10   #This is the encapsulant width in [cm]
    sn = 20     #This is the number of edge seal nodes to use
    en = 50     #This is the number of encapsulant nodes to use
    Esw = Esw/sn
    Encw = Encw/en

    return ingress_data