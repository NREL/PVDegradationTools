import pytest

from pvdeg import hpc

def test__chunksize_from_processes():

    res = hpc._chunksize_from_processes(
        n_proc=16,
        dataset_size=8e9, # 8GB
        max_chunk_size=2e8, # 200MB
    )

    pytest.approx(res, 166666666.666)
