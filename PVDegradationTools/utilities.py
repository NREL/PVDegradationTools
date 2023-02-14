import os
import pandas as pd
from rex import NSRDBX

def write_gids(nsrdb_fp, region='Colorado', region_col='state', out_fn='gids'):
    """
    Generate a .CSV file containing the GIDs for the spatial test range.
    The .CSV file will be saved to the working directory
    
    TODO: specify output file name and directory?

    Parameters:
    -----------
    nsrdb_fp : (str, path_obj)
        full file path to the NSRDB h5 file containing the weather data and GIDs
    region : (str, default = "Colorado")
        Name of the NSRDB region you are filtering into the GID list
    region_col : (str, default = "Sate")
        Name of the NSRDB region type
    out_fd : (str, default = "gids")
        Name of data column you want to retrieve. Generally, this should be "gids"

    Returns:
    -----------
    None
    """

    with NSRDBX(nsrdb_fp, hsds=False) as f:
        gids = f.region_gids(region=region, region_col=region_col)   

    df_gids = pd.DataFrame(gids, columns=['gid'])
    df_gids.to_csv('{}.csv'.format(out_fn), index=False)


def convert_tmy(file_in, file_out):
    '''
    Read a older TMY-like weather file and convert to h5 for use in PVD

    Parameters:
    -----------
    file_in : (str, path_obj)
        full file path to existing weather file
    file_out : (str, path_obj)
        full file path and name of file to create.
    '''
    pass