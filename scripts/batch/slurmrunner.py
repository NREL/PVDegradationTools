#       replaces Dask.jobqueue.SLURMRunner        #
###################################################
###################################################

import os
import getpass
import logging

from dask_jobqueue import SLURMCluster as SLURMRunner
from dask.distributed import Client as distributed_Client

logger = logging.getLogger(__name__)

logger.info("Getting dask cluster from SLURM.")
user_name = getpass.getuser()

proc_id = int(os.environ["SLURM_PROCID"])
n_workers = int(os.environ["SLURM_NTASKS"])  # n_tasks
job_id = int(os.environ["SLURM_JOB_ID"])
cpus_per_task = int(os.environ["SLURM_CPUS_PER_TASK"])
n_nodes = int(os.environ["SLURM_NNODES"])
mem_per_node = int(os.environ["SLURM_MEM_PER_NODE"])
mem_worker = mem_per_node / (n_workers * cpus_per_task) * 1e6

logger.info(f"Memory per worker: {mem_worker / 1e9} GB")

with SLURMRunner(
    scheduler_file=f"/scratch/{user_name}/scheduler-{job_id}.json",
    scheduler_options={
        # change to unique (so i dont overlap with martin)
        "dashboard_address": ":34567",
    },
    worker_options={
        "memory_limit": mem_worker,
        "local_directory": f"/scratch/{user_name}",  # faster IO nodes available
    },
) as runner:
    with distributed_Client(runner) as dask_client:
        dask_client.forward_logging("QA-runner")  # specify logger

        logger.info(f"Dask cluster dashboard at: {dask_client.dashboard_link}")
        logger.debug(f"Dask cluster client address: {dask_client.scheduler.address}")

        dask_client.wait_for_workers(runner.n_workers)
        logger.info("Successfully set up dask cluster.")

###################################################
###################################################

# call functions as defined in inspire.py


# PVFleets Martin Approach #
############################
# computation
# pass the client

# run_fleet(dask_client, arg)

# or can use current approach

# dask map blocks approach
# systems = []
# dask_client.submit(
# )
# systems.append()
