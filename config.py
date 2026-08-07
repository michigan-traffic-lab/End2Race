from pathlib import Path
from types import SimpleNamespace

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


PROJECT_CONFIG_SCHEMA = {
    "model": {
        "lidar_features",
        "speed_embedding_dim",
        "gru_hidden_size",
        "mlp_hidden_size",
        "speed_mask_probability",
        "lidar_normalization_k",
    },
    "training": {
        "batch_size",
        "learning_rate",
        "num_epochs",
        "speed_loss_weight",
        "gradient_clip_norm",
    },
    "vehicle": {
        "steering_limit",
        "maximum_speed",
        "wheelbase",
        "length",
        "width",
        "mass",
        "drag_coefficient",
        "gravity",
    },
    "expert": {
        "tracker_steps",
        "trajectory_points",
        "horizon",
        "time_step",
        "road_width",
        "road_step",
        "minimum_terminal_speed",
        "max_acceleration",
        "max_lateral_acceleration",
        "max_curvature",
        "hard_collision_distance",
        "soft_collision_clearance",
        "collision_cost_weight",
        "collision_cost_power",
        "min_lookahead",
        "max_lookahead",
        "lookahead_speed_scale",
        "steering_gain",
        "interpolation_points",
    },
}

RACETRACK_CONFIG_SCHEMA = {
    "raceline_generation": {
        "map_name",
        "map_image_extension",
        "num_lanes",
        "side_lane_center_shift_fraction",
        "clockwise",
        "inner_safe_distance",
        "outer_safe_distance",
        "num_laps",
        "preparation_step_size",
        "regularization_step_size",
        "interpolation_step_size",
        "smoothing_regularization",
        "smoothing_length",
        "dynamic_model_exponent",
        "velocity_filter_window",
    },
    "conversion": {"pattern", "output_suffix", "require_confirmation"},
    "greyscale": {"input_path", "output_path"},
    "rename": {
        "donkey_waypoint_suffix",
        "centerline_source_suffix",
        "centerline_target_suffix",
        "raceline_source_suffix",
        "raceline_target_suffix",
        "remove_extensions",
    },
}


def _validate_mapping(values, schema, location):
    if not isinstance(values, dict):
        raise ValueError(f"{location} must contain a YAML mapping")

    missing = set(schema) - set(values)
    unknown = set(values) - set(schema)
    if missing or unknown:
        raise ValueError(
            f"Invalid keys in {location}: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )

    if isinstance(schema, dict):
        for key, child_schema in schema.items():
            _validate_mapping(values[key], child_schema, f"{location}.{key}")


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(
            **{
                key: _to_namespace(item)
                for key, item in value.items()
            }
        )
    return value


def load_config(path, schema):
    config_path = PROJECT_ROOT / path
    with config_path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)

    _validate_mapping(values, schema, config_path)
    return _to_namespace(values)


def load_project_config():
    return load_config("config.yaml", PROJECT_CONFIG_SCHEMA)


def load_racetrack_config():
    return load_config("f1tenth_racetracks/config.yaml", RACETRACK_CONFIG_SCHEMA)


def merge_config_sections(*sections):
    values = {}
    for section in sections:
        overlap = values.keys() & vars(section).keys()
        if overlap:
            raise ValueError(f"Duplicate merged configuration keys: {sorted(overlap)}")
        values.update(vars(section))
    return SimpleNamespace(**values)
