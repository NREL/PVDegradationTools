import pandas as pd
from rex import NSRDBX

def write_gids(nsrdb_fp, region='Colorado', region_col='state', out_fn='gids'):

    """
    #TODO: add documentation
    """

    with NSRDBX(nsrdb_fp, hsds=False) as f:
        gids = f.region_gids(region=region, region_col=region_col)   

    df_gids = pd.DataFrame(gids, columns=['gid'])
    df_gids.to_csv('{}.csv'.format(out_fn), index=False)
