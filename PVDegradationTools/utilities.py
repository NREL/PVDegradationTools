import os
import pandas as pd
import numpy as np
from rex import NSRDBX, Outputs

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


def _get_state(id):
    """
    Returns the full name of a state based on two-letter state code
    
    Parameters:
    -----------
    id : (str)
        two letter state code (example: CO, AZ, MD)

    Returns:
    -----------
    state_name : (str)
        full name of US state (example: Colorado, Arizona, Maryland)
    """
    state_dict = {'AK': 'Alaska',
            'AL': 'Alabama',
            'AR': 'Arkansas',
            'AS': 'American Samoa',
            'AZ': 'Arizona',
            'CA': 'California',
            'CO': 'Colorado',
            'CT': 'Connecticut',
            'DC': 'District of Columbia',
            'DE': 'Delaware',
            'FL': 'Florida',
            'GA': 'Georgia',
            'GU': 'Guam',
            'HI': 'Hawaii',
            'IA': 'Iowa',
            'ID': 'Idaho',
            'IL': 'Illinois',
            'IN': 'Indiana',
            'KS': 'Kansas',
            'KY': 'Kentucky',
            'LA': 'Louisiana',
            'MA': 'Massachusetts',
            'MD': 'Maryland',
            'ME': 'Maine',
            'MI': 'Michigan',
            'MN': 'Minnesota',
            'MO': 'Missouri',
            'MP': 'Northern Mariana Islands',
            'MS': 'Mississippi',
            'MT': 'Montana',
            'NA': 'National',
            'NC': 'North Carolina',
            'ND': 'North Dakota',
            'NE': 'Nebraska',
            'NH': 'New Hampshire',
            'NJ': 'New Jersey',
            'NM': 'New Mexico',
            'NV': 'Nevada',
            'NY': 'New York',
            'OH': 'Ohio',
            'OK': 'Oklahoma',
            'OR': 'Oregon',
            'PA': 'Pennsylvania',
            'PR': 'Puerto Rico',
            'RI': 'Rhode Island',
            'SC': 'South Carolina',
            'SD': 'South Dakota',
            'TN': 'Tennessee',
            'TX': 'Texas',
            'UT': 'Utah',
            'VA': 'Virginia',
            'VI': 'Virgin Islands',
            'VT': 'Vermont',
            'WA': 'Washington',
            'WI': 'Wisconsin',
            'WV': 'West Virginia',
            'WY': 'Wyoming'}
    state_name = state_dict[id]
    return state_name
    
def convert_tmy(file_in, file_out='h5_from_tmy.h5'):
    '''
    Read a older TMY-like weather file and convert to h5 for use in PVD

    TODO: figure out scale_facator and np.int32 for smaller file

    Parameters:
    -----------
    file_in : (str, path_obj)
        full file path to existing weather file
    file_out : (str, path_obj)
        full file path and name of file to create.
    '''
    from pvlib import iotools

    src_data, src_meta = iotools.tmy.read_tmy3(file_in)

    save_cols = {'DNI':'dni',
            'DHI':'dhi',
            'GHI':'ghi',
            'DryBulb':'air_temperature',
            'DewPoint':'dew_point',
            'RHum':'relative_humidity',
            'Wspd':'wind_speed',
            'Alb':'albedo'}

    df_new = src_data[save_cols.keys()].copy()
    df_new.columns = save_cols.values()
    time_index = df_new.index

    meta = {'latitude':[src_meta['latitude']],
            'longitude':[src_meta['longitude']],
            'elevation':[src_meta['altitude']],
            'timezone':[src_meta['TZ']],
            'country':['United States'],
            'state':[_get_state(src_meta['State'])]}
    meta = pd.DataFrame(meta)

    with Outputs(file_out, 'w') as f:
        f.meta = meta
        f.time_index = time_index
    
    for col in df_new.columns:
        Outputs.add_dataset(h5_file=file_out, dset_name=col,
                            dset_data=df_new[col].values,
                            attrs={'scale_factor':100},
                            dtype=np.int64)
