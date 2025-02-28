"""
High Performance Computing (HPC) utilities for parallel/distributed computing
"""

import numpy as np
import xarray as xr


def _chunksize_from_processes(n_proc: int, dataset_size: int, max_chunk_size: int = 2e8):
    """
    Parameters
    ----------
    n_proc: int
        number of processes

    dataset_size: int
        size of dataset in bytes

    max_chunk_size: int
        maximum chunk size in bytes. default 2e8 bytes = 200 MB

    Returns
    --------
    Target task size in bytes. In this case "Task" means combination of chunks that will be evaluated by the same worker.
    """

    min_num_tasks = int(np.ceil(dataset_size / max_chunk_size))
    print(f"lambda (min_num_tasks): {min_num_tasks}")

    sigma = -1

    for i in range(min_num_tasks, -1, -1):
        if i % n_proc == 0:
            sigma = i
            break

    if sigma == -1:
        raise ValueError("no suitable number of processes found")

    print("sigma", sigma)

    good_tasks_num = sigma + n_proc
    print("good_tasks_num", good_tasks_num)

    target_task_size = dataset_size / good_tasks_num
    print("target task size [B]", target_task_size)
    print("target task size [MB]", target_task_size / 1e6)

    return target_task_size


def chunk_via_processes(ds: xr.Dataset, n_proc: int, max_chunk_size: int = 2e8):
    # i belive we only are care about the size of the datavars, as we are only chunking these
    data_vars_size_bytes = sum(var.nbytes for var in ds.data_vars.values())

    target_task_size = _chunksize_from_processes(
        n_proc=n_proc, max_chunk_size=max_chunk_size, dataset_size=data_vars_size_bytes
    )

    single_location_single_time_size = sum([ds[var].dtype.itemsize for var in ds.data_vars])

    single_location_size = single_location_single_time_size * ds.sizes["time"]

    num_locations_per_task = target_task_size / single_location_size

    return ds.chunk({"gid": num_locations_per_task, "time": -1})