from pvdeg.standards import run_calc_standoff
from pvdeg.humidity import run_module
from gaps.cli import CLICommandFromFunction, make_cli

commands = [
    CLICommandFromFunction(
        name="run-standoff", function=run_calc_standoff, split_keys=["project_points"]
    ),
    CLICommandFromFunction(
        name="run-rel_humidity", function=run_module, split_keys=["project_points"]
    ),
]
cli = make_cli(commands)


if __name__ == "__main__":
    cli(obj={})
