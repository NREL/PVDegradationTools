import pvdeg

import sys

import pandas as pd

from dask.distributed import Client
from dask_jobqueue import SLURMCluster


cluster = SLURMCluster(
    queue="shared",
    account="inspire",
    cores=1,
    memory="30 GB",
    processes=True,
    log_directory="/scratch/tford/dev/logs",
    walltime="02:00:00",
)
cluster.scale(64)

client = Client(cluster)

print(client.dashboard_link)

locationGetter = pvdeg.scenario.GeospatialScenario()
locationGetter.addLocation(
    country="United States",
    state="CO",
    downsample_factor=2,
    nsrdb_attributes=pvdeg.pysam.INSPIRE_NSRDB_ATTRIBUTES,
)
geo_weather, geo_meta = locationGetter.geospatial_data()

# smaller chunks to allow the job to be splitup between all of the workers
# each worker should have at least one chunk
geo_weather = geo_weather.chunk({"gid": 2})

shapes = {
    "annual_poa": ("gid",),
    "annual_energy": ("gid",),
    "poa_front": (
        "gid",
        "time",
    ),
    "poa_rear": (
        "gid",
        "time",
    ),
    "subarray1_poa_front": ("gid", "time"),
    "subarray1_poa_rear": ("gid", "time"),
    "ground_irradiance": ("gid", "time", "distance"),  # spatio-temporal
}

template = pvdeg.geospatial.output_template(
    ds_gids=geo_weather,  # times will cause error, fix below
    shapes=shapes,
    add_dims={
        "distance": 10
    },  # this will autogenerate a range of length 10 for the coordinate axis
)

# modified range as produced by the corrected times for the tmy dataset
template["time"] = pd.date_range(start="2001-01-01 00:30:00", freq="1h", periods=8760)

divisibleby = 3
step = geo_meta.shape[0] // divisibleby


# conf = "01"
conf = sys.argv[1]
target_dir = f"/projects/inspire/PySAM-MAPS/CO-sample/{conf}"

for i in range(0, geo_meta.shape[0], step):
    print("started", i)

    front, back = i, i + step

    slice_weather = geo_weather.isel(gid=slice(front, back))
    slice_meta = geo_meta.iloc[front:back]
    slice_template = template.isel(gid=slice(front, back))

    inspire_partial_res = pvdeg.geospatial.analysis(
        weather_ds=slice_weather,
        meta_df=slice_meta,
        template=slice_template,
        func=pvdeg.pysam.inspire_ground_irradiance,
        config_files={
            "pv": f"/home/tford/dev/InSPIRE/Studies/USMap_Doubleday_2024/SAM/{conf}/{conf}_pvsamv1.json"  # noqa
        },
    )

    inspire_partial_res.to_netcdf(
        f"{target_dir}-quarter-res-{i}-{i + i - 1}.nc", engine="h5netcdf"
    )
    print("ended", i)
