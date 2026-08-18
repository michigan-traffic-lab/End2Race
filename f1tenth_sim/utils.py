from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


def load_yaml_config(path):
    """Load a config whose sections become nested namespaces."""
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)

    return SimpleNamespace(**{name: SimpleNamespace(**section) for name, section in values.items()})


def racetrack_path(*parts):
    """Path to a file inside the bundled racetrack collection."""
    return Path(__file__).resolve().parent / "f1tenth_racetracks" / Path(*parts)


def load_racetrack_config():
    return load_yaml_config(racetrack_path("config.yaml"))


def load_simulation_config():
    return load_yaml_config(Path(__file__).resolve().parent / "config.yaml")


def simulation_config():
    """Timing contract and initial ego speed derived from the simulator configuration."""
    simulation = load_simulation_config().simulation
    frequency = simulation.frequency_hz
    control_frequency = simulation.control_frequency_hz
    planner_frequency = simulation.expert_planner_frequency_hz
    # Integer step counts truncate silently unless the frequencies divide evenly
    if frequency % control_frequency or frequency % planner_frequency:
        raise ValueError(
            "simulation.frequency_hz must divide both simulation.control_frequency_hz "
            "and simulation.expert_planner_frequency_hz"
        )
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
    if values.shape[1] < 6:
        raise ValueError(f"{path} must contain at least six columns")
    return values[:, [1, 2, 3, 5]]


def load_raceline_start(map_name, raceline_file, start_idx):
    waypoints = load_raceline(map_name, raceline_file)
    idx = start_idx % len(waypoints)
    start_pose = np.array(
        [[waypoints[idx, 0], waypoints[idx, 1], waypoints[idx, 2]]]
    )
    return start_pose, waypoints
