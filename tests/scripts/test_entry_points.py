from importlib.metadata import entry_points

import pytest

from lerobot_policy_robomimic.scripts import convert_checkpoint, convert_dataset


@pytest.mark.parametrize(
    ("command", "main"),
    [
        ("robomimic-convert-dataset", convert_dataset.main),
        ("robomimic-convert-checkpoint", convert_checkpoint.main),
    ],
)
def test_command_runs_the_converter(command, main):
    (entry_point,) = entry_points(group="console_scripts", name=command)

    assert entry_point.load() is main
