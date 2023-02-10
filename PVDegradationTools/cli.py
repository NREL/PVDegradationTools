from PVDegradationTools.standards import run_standoff
from PVDegradationTools.humidity import run_rel_humidity
from gaps.cli import CLICommandConfiguration, make_cli

commands = [
    CLICommandConfiguration(
        name="run-standoff", 
        function=run_standoff, 
        split_keys=["project_points"]
    ),
    CLICommandConfiguration(
        name="run-rel_humidity", 
        function=run_rel_humidity, 
        split_keys=["project_points"]
    ),
]
cli = make_cli(commands)


if __name__ == "__main__":
    cli(obj={})