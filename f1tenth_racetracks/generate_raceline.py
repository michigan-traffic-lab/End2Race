import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import trajectory_planning_helpers as tph
import yaml
from scipy import interpolate

from config import (
    load_project_config,
    load_racetrack_config,
    merge_config_sections,
)


def spline_distance(t_glob, path, point):
    """Return a scalar spline distance for SciPy optimizer callbacks."""
    parameter = float(np.asarray(t_glob).reshape(-1)[0])
    spline_point = np.asarray(interpolate.splev(parameter, path))
    return float(np.linalg.norm(point - spline_point))


def prepare_track(reference_track, config):
    """Smooth a lane and calculate its normalized spline vectors."""
    tph.spline_approximation.dist_to_p = spline_distance
    track = tph.spline_approximation.spline_approximation(
        track=reference_track,
        k_reg=config.smoothing_regularization,
        s_reg=config.smoothing_length,
        stepsize_prep=config.preparation_step_size,
        stepsize_reg=config.regularization_step_size,
        debug=False,
    )
    closed_path = np.vstack((track[:, :2], track[0, :2]))
    _, _, _, normal_vectors = tph.calc_splines.calc_splines(path=closed_path)

    normals_crossing = tph.check_normals_crossing.check_normals_crossing(
        track=track,
        normvec_normalized=normal_vectors,
        horizon=10,
    )

    if normals_crossing:
        outer_bound = track[:, :2] + normal_vectors * track[:, 2, None]
        inner_bound = track[:, :2] - normal_vectors * track[:, 3, None]

        plt.figure()
        plt.plot(track[:, 0], track[:, 1], "k-")
        for index in range(len(outer_bound)):
            boundary = np.vstack((outer_bound[index], inner_bound[index]))
            plt.plot(boundary[:, 0], boundary[:, 1], "r-", linewidth=0.7)

        plt.grid()
        plt.gca().set_aspect("equal", "datalim")
        plt.xlabel("east in m")
        plt.ylabel("north in m")
        plt.title("Error: at least one pair of normals is crossed!")
        plt.show()
        raise RuntimeError(
            "At least two spline normals cross; check the input or increase "
            "the smoothing factor"
        )

    minimum_width = 2.0 * config.width
    widths = track[:, 2] + track[:, 3]
    narrow = widths < minimum_width
    if np.any(narrow):
        inflation = 0.5 * (minimum_width - widths[narrow])
        track[narrow, 2] += inflation
        track[narrow, 3] += inflation

        print(
            "Track was narrower than the vehicle requirement; inflated both "
            "boundaries equally",
            file=sys.stderr,
        )

    return track, normal_vectors


def reorder_vertex(image, lane):
    """Reorder vertices to form a continuous path."""
    path_img = np.zeros_like(image)
    for point in lane:
        cv2.circle(path_img, point, 1, (255, 255, 255), 1)
    kernel = np.ones((2, 2), np.uint8)
    dilation_count = 0
    while True:
        if dilation_count > 10:
            raise RuntimeError("Could not connect the sampled lane vertices")
        contours, hierarchy = cv2.findContours(
            path_img,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) == 2 and hierarchy[0][-1][-1] == 0:
            break
        path_img = cv2.dilate(path_img, kernel, iterations=1)
        dilation_count += 1
    path_img = cv2.ximgproc.thinning(path_img)
    contours, _ = cv2.findContours(
        path_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return np.squeeze(contours[0])


def transform_coords(path, height, scale, offset_x, offset_y):
    """Transform pixel coordinates to world coordinates."""
    new_path_x = path[:, 0] * scale + offset_x
    new_path_y = (height - path[:, 1]) * scale + offset_y
    if path.shape[1] > 2:
        return np.column_stack(
            (
                new_path_x,
                new_path_y,
                path[:, 2] * scale,
                path[:, 3] * scale,
            )
        )
    return np.column_stack((new_path_x, new_path_y))


def generate_lanes(config, map_dir):
    """Generate lanes from map image."""
    yaml_file = map_dir / f"{config.map_name}_map.yaml"
    with yaml_file.open(encoding="utf-8") as stream:
        parsed_yaml = yaml.safe_load(stream)
    scale = parsed_yaml["resolution"]
    offset_x = parsed_yaml["origin"][0]
    offset_y = parsed_yaml["origin"][1]

    shift_fraction = config.side_lane_center_shift_fraction
    if not 0.0 <= shift_fraction < 1.0:
        raise ValueError(
            "side_lane_center_shift_fraction must be in [0, 1)"
        )

    lane_fractions = (
        np.arange(1, config.num_lanes + 1) / (config.num_lanes + 1)
    )
    lane_fractions += (0.5 - lane_fractions) * shift_fraction
    lane_ratios = lane_fractions / (1.0 - lane_fractions)

    image_path = map_dir / (
        f"{config.map_name}_map{config.map_image_extension}"
    )
    input_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    h, w = input_img.shape[:2]

    output_img = ~input_img
    _, output_img = cv2.threshold(
        output_img,
        thresh=127,
        maxval=255,
        type=cv2.THRESH_BINARY,
    )

    contours, _ = cv2.findContours(
        output_img,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    for contour in contours:
        if cv2.contourArea(contour) < 70:
            cv2.fillPoly(output_img, pts=[contour], color=(0, 0, 0))

    kernel = np.ones((5, 5), np.uint8)
    output_img = cv2.dilate(output_img, kernel, iterations=1)
    output_img = cv2.ximgproc.thinning(output_img)

    contours, hierarchy = cv2.findContours(
        output_img,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    parents = hierarchy[0][:, 3]

    node = np.argmax(parents)
    tree_indices = []
    while node != -1:
        tree_indices.append(node)
        node = parents[node]
    tree_indices.reverse()

    outer_bound = contours[tree_indices[1]]
    inner_bound = contours[tree_indices[2]]

    x_coordinates, y_coordinates = np.meshgrid(np.arange(w), np.arange(h))
    valid_pts = []
    for x, y in zip(x_coordinates.ravel(), y_coordinates.ravel()):
        point = (int(x), int(y))
        outer_dist = cv2.pointPolygonTest(outer_bound, point, True)
        inner_dist = cv2.pointPolygonTest(inner_bound, point, True)
        if (
            outer_dist > config.outer_safe_distance / scale
            and inner_dist < -config.inner_safe_distance / scale
        ):
            ratio = np.abs(inner_dist) / (np.abs(outer_dist) + 1e-8)
            valid_pts.append([x, y, inner_dist, outer_dist, ratio])

    valid_pts = np.array(valid_pts)

    lanes = []
    for index, lane_ratio in enumerate(lane_ratios):
        valid_ratio = (
            np.abs(valid_pts[:, -1] - lane_ratio) < lane_ratio / 10
        )
        lane = valid_pts[valid_ratio, 0:2].astype(int)
        lane = reorder_vertex(output_img, lane)
        if config.clockwise:
            lane = np.flipud(lane)

        left_dists, right_dists = [], []
        for x, y in lane:
            point = (int(x), int(y))
            outer_dist = cv2.pointPolygonTest(outer_bound, point, True)
            inner_dist = cv2.pointPolygonTest(inner_bound, point, True)
            outer_dist = outer_dist - config.outer_safe_distance / scale
            inner_dist = abs(inner_dist) - config.inner_safe_distance / scale
            if config.clockwise:
                left_dists.append(outer_dist)
                right_dists.append(inner_dist)
            else:
                left_dists.append(inner_dist)
                right_dists.append(outer_dist)

        lane = np.vstack((lane.T, right_dists, left_dists)).T
        lane = transform_coords(lane, h, scale, offset_x, offset_y)
        csv_path = map_dir / f"lane{index}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["#x_m", "y_m", "w_tr_right_m", "w_tr_left_m"])
            writer.writerows(lane)
        lanes.append(lane)

    return lanes


def generate_raceline(lane_data, config, module):
    """Generate raceline trajectory for a given lane."""
    dynamics_directory = module / "vehicle_dynamic_info"
    ggv, ax_max_machines = tph.import_veh_dyn_info.import_veh_dyn_info(
        ggv_import_path=str(dynamics_directory / "ggv.csv"),
        ax_max_machines_import_path=str(
            dynamics_directory / "ax_max_machines.csv"
        ),
    )

    max_ggv_velocity = np.max(ggv[:, 0])
    maximum_speed = config.maximum_speed
    if maximum_speed > max_ggv_velocity:
        maximum_speed = max_ggv_velocity * 0.95

    reference_track, normal_vectors = prepare_track(lane_data, config)
    lateral_offsets = np.zeros(len(reference_track))

    (
        raceline,
        _,
        x_coefficients,
        y_coefficients,
        spline_indices,
        spline_parameters,
        distances,
        spline_lengths,
        element_lengths,
    ) = tph.create_raceline.create_raceline(
        refline=reference_track[:, :2],
        normvectors=normal_vectors,
        alpha=lateral_offsets,
        stepsize_interp=config.interpolation_step_size,
    )

    headings, curvatures = tph.calc_head_curv_an.calc_head_curv_an(
        coeffs_x=x_coefficients,
        coeffs_y=y_coefficients,
        ind_spls=spline_indices,
        t_spls=spline_parameters,
    )

    velocity_profile = tph.calc_vel_profile.calc_vel_profile(
        ggv=ggv,
        ax_max_machines=ax_max_machines,
        v_max=maximum_speed,
        kappa=curvatures,
        el_lengths=element_lengths,
        closed=True,
        filt_window=config.velocity_filter_window,
        dyn_model_exp=config.dynamic_model_exponent,
        drag_coeff=config.drag_coefficient,
        m_veh=config.mass,
    )

    closed_velocity_profile = np.append(velocity_profile, velocity_profile[0])
    acceleration_profile = tph.calc_ax_profile.calc_ax_profile(
        vx_profile=closed_velocity_profile,
        el_lengths=element_lengths,
        eq_length_output=False,
    )

    time_profile = tph.calc_t_profile.calc_t_profile(
        vx_profile=velocity_profile,
        ax_profile=acceleration_profile,
        el_lengths=element_lengths,
    )

    trajectory = np.column_stack(
        (
            distances,
            raceline,
            headings + 0.5 * np.pi,
            curvatures,
            velocity_profile,
            acceleration_profile,
        )
    )
    closed_trajectory = np.vstack((trajectory, trajectory[0]))
    closed_trajectory[-1, 0] = np.sum(spline_lengths)
    return closed_trajectory, time_profile[-1]


def main():
    module = Path(__file__).resolve().parent
    project = load_project_config()
    racetrack = load_racetrack_config()
    config = merge_config_sections(
        racetrack,
        project.vehicle,
    )
    map_dir = module / config.map_name
    map_dir.mkdir(exist_ok=True)

    print(f"Generating lanes for {config.map_name}...")
    lanes = generate_lanes(config, map_dir)

    for index, lane in enumerate(lanes):
        lane_name = f"lane{index}"
        print(f"\nProcessing {lane_name}...")
        trajectory, laptime = generate_raceline(lane, config, module)
        export_path = map_dir / f"raceline{index}.csv"
        header = "s_m;x_m;y_m;psi_rad;kappa_radpm;vx_mps;ax_mps2"
        np.savetxt(
            export_path,
            trajectory,
            delimiter=";",
            fmt="%.6f",
            header=header,
        )

        print(f"  Estimated laptime: {laptime:.2f}s")
        print(f"  Trajectory exported to: {export_path}")


if __name__ == "__main__":
    main()
