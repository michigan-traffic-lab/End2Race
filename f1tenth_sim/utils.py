from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

SIMULATOR_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_simulator_config():
    with SIMULATOR_CONFIG_PATH.open(encoding="utf-8") as stream:
        sections = yaml.safe_load(stream)
    return SimpleNamespace(**{
        name: SimpleNamespace(**values) for name, values in sections.items()
    })


def racetrack_path(*parts):
    """Path to a file inside the bundled racetrack collection."""
    return Path(__file__).resolve().parent / "f1tenth_racetracks" / Path(*parts)


def simulation_config():
    """Timing contract and initial ego speed derived from the simulator configuration."""
    simulation = load_simulator_config().simulation
    frequency = simulation.frequency_hz
    control_frequency = simulation.control_frequency_hz
    planner_frequency = simulation.expert_planner_frequency_hz
    return SimpleNamespace(
        frequency_hz=frequency,
        control_frequency_hz=control_frequency,
        expert_planner_frequency_hz=planner_frequency,
        steps_per_control=frequency // control_frequency,
        steps_per_expert_plan=frequency // planner_frequency,
        timestep=1.0 / frequency,
        control_timestep=1.0 / control_frequency,
        video_fps=frequency,
        ego_initial_speed_fraction=simulation.ego_initial_speed_fraction,
    )


def load_raceline(map_name, raceline_file):
    """Load x, y, heading, and speed columns from a raceline."""
    path = racetrack_path(map_name, raceline_file)
    values = np.loadtxt(path, delimiter=";", skiprows=1, ndmin=2)
    return values[:, [1, 2, 3, 5]]
