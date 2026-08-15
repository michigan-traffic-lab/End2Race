from dataclasses import dataclass
import numpy as np
from utils import find_opponent_start_index, get_ego_idx_range, load_raceline


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    ego_idx: int
    opponent_idx: int
    opponent_raceline: str
    opponent_speed_scale: float


def build_scenario_pool(simulation):
    """Build the ego-start x opponent-raceline x speed-scale matrix."""
    ego_waypoints = load_raceline(simulation.map_name, f"{simulation.ego_raceline}.csv")
    ego_indices = get_ego_idx_range(simulation.map_name, simulation.ego_raceline, simulation.num_startpoints)

    scenarios = []
    for opponent_raceline in simulation.opponent_racelines:
        if opponent_raceline == simulation.ego_raceline:
            opponent_waypoints = ego_waypoints
        else:
            opponent_waypoints = load_raceline(simulation.map_name, f"{opponent_raceline}.csv")
        for opponent_speed_scale in simulation.opponent_speed_scales:
            for ego_idx in ego_indices:
                opponent_idx = find_opponent_start_index(ego_waypoints, opponent_waypoints, ego_idx, simulation.interval_index)
                scenario_id = f"e{ego_idx:04d}_{opponent_raceline}_s{opponent_speed_scale}"
                scenarios.append(Scenario(scenario_id, int(ego_idx), int(opponent_idx), opponent_raceline, float(opponent_speed_scale)))
    return tuple(scenarios)


def scenario_stream(scenarios, seed):
    """Yield scenarios in repeated no-replacement passes over the pool."""
    rng = np.random.default_rng(seed)
    while True:
        for index in rng.permutation(len(scenarios)):
            yield scenarios[int(index)]
