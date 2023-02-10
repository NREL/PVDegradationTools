"""Collection of classes and functions for humidity calculations.
"""

import numpy as np
import pandas as pd
import pvlib
from rex import NSRDBX
from rex import Outputs
from pathlib import Path
from random import random
from concurrent.futures import ProcessPoolExecutor, as_completed


#TODO: Generic dummy function - examples for Matt
def calc_rel_humidity(nsrdb_fp, gid):

    "generate random numbers for rel humidity dummy run"

    return {'RH_front_encap' : random(), 
            'RH_back_encap' : random(), 
            'RH_backsheet' : random()}


def run_rel_humidity(
    project_points, 
    out_dir, 
    tag,
    nsrdb_fp,
    max_workers=None
):

    "dummy run to showcase multiple commands in cli.py"

    all_fields = ['RH_front_encap', 'RH_back_encap', 'RH_backsheet']

    with NSRDBX(nsrdb_fp, hsds=False) as f:
        meta = f.meta[f.meta.index.isin(project_points.gids)]
        ti = f.time_index

    out_fp = Path(out_dir) / f"out_rel_hum{tag}.h5"
    shapes = {n : (len(ti), len(project_points)) for n in all_fields}
    attrs = {n : None for n in all_fields}
    chunks = {n : None for n in all_fields}
    dtypes = {n : "float32" for n in all_fields}

    Outputs.init_h5(
        out_fp,
        all_fields,
        shapes,
        attrs,
        chunks,
        dtypes,
        meta=meta.reset_index(),
        time_index=ti
    )

    future_to_point = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for point in project_points:
            gid = int(point.gid)
            future = executor.submit(
                calc_rel_humidity,
                nsrdb_fp, 
                gid, 
            )
            future_to_point[future] = gid

        with Outputs(out_fp, mode="a") as out:
            for future in as_completed(future_to_point):
                result = future.result()
                gid = future_to_point.pop(future)

                ind = project_points.index(gid)
                for dset, data in result.items():
                    out[dset, :, ind] = np.repeat(data, len(ti))

    return out_fp.as_posix()