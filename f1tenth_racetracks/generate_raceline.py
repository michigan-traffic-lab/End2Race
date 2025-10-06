import cv2
import numpy as np
import os
import argparse
import yaml
import numpy as np
import sys
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate lanes and raceline trajectories')
    
    # Map parameters
    parser.add_argument('--map_name', type=str, default='Shanghai',
                       help='Name of the map to process')
    parser.add_argument('--map_img_ext', type=str, default='.png',
                       help='Image file extension')
    parser.add_argument('--num_lanes', type=int, default=1,
                       help='Number of lanes to generate')
    parser.add_argument('--clockwise', action='store_true', default=True,
                       help='Track direction is clockwise')
    parser.add_argument('--inner_safe_dist', type=float, default=0.4,
                       help='Safety distance from inner boundary (meters)')
    parser.add_argument('--outer_safe_dist', type=float, default=0.4,
                       help='Safety distance from outer boundary (meters)')
    parser.add_argument('--opp_safe_dist', type=float, default=0.25,
                       help='Safety distance for opponent detection (meters)')
    
    # Vehicle parameters
    parser.add_argument('--v_max', type=float, default=7.5,
                       help='Maximum velocity in m/s')
    parser.add_argument('--vehicle_length', type=float, default=0.51,
                       help='Vehicle length in meters')
    parser.add_argument('--vehicle_width', type=float, default=0.31,
                       help='Vehicle width in meters')
    parser.add_argument('--vehicle_mass', type=float, default=3.362,
                       help='Vehicle mass in kg')
    parser.add_argument('--drag_coeff', type=float, default=0.0075,
                       help='Drag coefficient')
    parser.add_argument('--num_laps', type=int, default=2,
                       help='Number of laps for trajectory')
    
    return parser.parse_args()

def prep_track(reftrack_imp: np.ndarray,
               reg_smooth_opts: dict,
               stepsize_opts: dict,
               debug: bool = True,
               min_width: float = None) -> tuple:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    This function prepares the inserted reference track for optimization.

    Inputs:
    reftrack_imp:               imported track [x_m, y_m, w_tr_right_m, w_tr_left_m]
    reg_smooth_opts:            parameters for the spline approximation
    stepsize_opts:              dict containing the stepsizes before spline approximation and after spline interpolation
    debug:                      boolean showing if debug messages should be printed
    min_width:                  [m] minimum enforced track width (None to deactivate)

    Outputs:
    reftrack_interp:            track after smoothing and interpolation [x_m, y_m, w_tr_right_m, w_tr_left_m]
    normvec_normalized_interp:  normalized normal vectors on the reference line [x_m, y_m]
    a_interp:                   LES coefficients when calculating the splines
    coeffs_x_interp:            spline coefficients of the x-component
    coeffs_y_interp:            spline coefficients of the y-component
    """

    # smoothing and interpolating reference track
    reftrack_interp = tph.spline_approximation. \
        spline_approximation(track=reftrack_imp,
                             k_reg=reg_smooth_opts["k_reg"],
                             s_reg=reg_smooth_opts["s_reg"],
                             stepsize_prep=stepsize_opts["stepsize_prep"],
                             stepsize_reg=stepsize_opts["stepsize_reg"],
                             debug=debug)

    # calculate splines
    refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))

    coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = tph.calc_splines.\
        calc_splines(path=refpath_interp_cl)


    normals_crossing = tph.check_normals_crossing.check_normals_crossing(track=reftrack_interp,
                                                                         normvec_normalized=normvec_normalized_interp,
                                                                         horizon=10)

    if normals_crossing:
        bound_1_tmp = reftrack_interp[:, :2] + normvec_normalized_interp * np.expand_dims(reftrack_interp[:, 2], axis=1)
        bound_2_tmp = reftrack_interp[:, :2] - normvec_normalized_interp * np.expand_dims(reftrack_interp[:, 3], axis=1)

        plt.figure()

        plt.plot(reftrack_interp[:, 0], reftrack_interp[:, 1], 'k-')
        for i in range(bound_1_tmp.shape[0]):
            temp = np.vstack((bound_1_tmp[i], bound_2_tmp[i]))
            plt.plot(temp[:, 0], temp[:, 1], "r-", linewidth=0.7)

        plt.grid()
        ax = plt.gca()
        ax.set_aspect("equal", "datalim")
        plt.xlabel("east in m")
        plt.ylabel("north in m")
        plt.title("Error: at least one pair of normals is crossed!")

        plt.show()

        raise IOError("At least two spline normals are crossed, check input or increase smoothing factor!")

    manipulated_track_width = False

    if min_width is not None:
        for i in range(reftrack_interp.shape[0]):
            cur_width = reftrack_interp[i, 2] + reftrack_interp[i, 3]

            if cur_width < min_width:
                manipulated_track_width = True

                # inflate to both sides equally
                reftrack_interp[i, 2] += (min_width - cur_width) / 2
                reftrack_interp[i, 3] += (min_width - cur_width) / 2

    if manipulated_track_width:
        print("WARNING: Track region was smaller than requested minimum track width -> Applied artificial inflation in"
              " order to match the requirements!", file=sys.stderr)

    return reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp


def reorder_vertex(image, lane):
    """Reorder vertices to form a continuous path."""
    path_img = np.zeros_like(image)
    for idx in range(len(lane)):
        cv2.circle(path_img, lane[idx], 1, (255, 255, 255), 1)
    curr_kernel = np.ones((2, 2), np.uint8)
    iter_cnt = 0
    while True:
        if iter_cnt > 10:
            exit(0)
        curr_contours, curr_hierarchy = cv2.findContours(path_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(curr_contours) == 2 and curr_hierarchy[0][-1][-1] == 0:
            break
        path_img = cv2.dilate(path_img, curr_kernel, iterations=1)
        iter_cnt += 1
    path_img = cv2.ximgproc.thinning(path_img)
    curr_contours, curr_hierarchy = cv2.findContours(path_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return np.squeeze(curr_contours[0])


def transform_coords(path, height, s, tx, ty):
    """Transform pixel coordinates to world coordinates."""
    new_path_x = path[:, 0] * s + tx
    new_path_y = (height - path[:, 1]) * s + ty
    if path.shape[1] > 2:
        new_right_dist = path[:, 2] * s
        new_left_dist = path[:, 3] * s
        return np.vstack((new_path_x, new_path_y, new_right_dist, new_left_dist)).T
    else:
        return np.vstack((new_path_x, new_path_y)).T


def save_csv(data, csv_name, header=None):
    """Save data to CSV file."""
    import csv
    with open(csv_name, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if header:
            csv_writer.writerow(header)
        for line in data:
            csv_writer.writerow(line.tolist())

def generate_lanes(args, map_dir):
    """Generate lanes from map image."""
    # Read map parameters
    yaml_file = os.path.join(map_dir, args.map_name + "_map.yaml")
    with open(yaml_file, 'r') as stream:
        parsed_yaml = yaml.safe_load(stream)
    scale = parsed_yaml["resolution"]
    offset_x = parsed_yaml["origin"][0]
    offset_y = parsed_yaml["origin"][1]

    # Define lane ratios
    lane_ratios = np.arange(1, args.num_lanes + 1) / np.arange(args.num_lanes, 0, -1)
    if not np.any(lane_ratios == 1.0):
        lane_ratios = np.append(lane_ratios, 1.0)

    # Read image
    img_path = os.path.join(map_dir, args.map_name + "_map" + args.map_img_ext)
    input_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = input_img.shape[:2]

    # Process image
    output_img = ~input_img
    ret, output_img = cv2.threshold(output_img, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

    # Find and filter contours
    contours, hierarchy = cv2.findContours(output_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 70:
            cv2.fillPoly(output_img, pts=[contour], color=(0, 0, 0))

    # Dilate & Erode
    kernel = np.ones((5, 5), np.uint8)
    output_img = cv2.dilate(output_img, kernel, iterations=1)
    output_img = cv2.ximgproc.thinning(output_img)

    # Separate outer and inner bounds
    contours, hierarchy = cv2.findContours(output_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    parents = hierarchy[0][:, 3]
    
    node = np.argmax(parents)
    tree_indices = []
    while node != -1:
        tree_indices.append(node)
        node = parents[node]
    tree_indices.reverse()

    outer_bound = contours[tree_indices[1]]
    inner_bound = contours[tree_indices[2]]

    # Euclidean distance transform
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    X = X.flatten().tolist()
    Y = Y.flatten().tolist()
    valid_pts = []
    
    for (x, y) in zip(X, Y):
        outer_dist = cv2.pointPolygonTest(outer_bound, (x, y), True)
        inner_dist = cv2.pointPolygonTest(inner_bound, (x, y), True)
        if outer_dist > args.outer_safe_dist / scale and inner_dist < -args.inner_safe_dist / scale:
            ratio = np.abs(inner_dist) / (np.abs(outer_dist) + 1e-8)
            valid_pts.append([x, y, inner_dist, outer_dist, ratio])
    
    valid_pts = np.array(valid_pts)

    # Calculate lanes
    lanes = []
    lane_names = []
    for idx in range(len(lane_ratios)):
        valid_ratio = (np.abs(valid_pts[:, -1] - lane_ratios[idx]) < lane_ratios[idx] / 10)
        lane = valid_pts[valid_ratio, 0:2].astype(int)
        lane = reorder_vertex(output_img, lane)
        if args.clockwise:
            lane = np.flipud(lane)
        
        left_dists, right_dists = [], []
        for (x, y) in lane:
            outer_dist = cv2.pointPolygonTest(outer_bound, (int(x), int(y)), True)
            inner_dist = cv2.pointPolygonTest(inner_bound, (int(x), int(y)), True)
            outer_dist = outer_dist - args.outer_safe_dist / scale
            inner_dist = abs(inner_dist) - args.inner_safe_dist / scale
            if args.clockwise:
                left_dists.append(outer_dist)
                right_dists.append(inner_dist)
            else:
                left_dists.append(inner_dist)
                right_dists.append(outer_dist)
        
        lane = np.vstack((lane.T, right_dists, left_dists)).T
        # Transform to world coordinates
        lane = transform_coords(lane, h, scale, offset_x, offset_y)
        
        # Use unified naming: lane0, lane1, lane2, etc.
        lane_name = f"lane{idx}"
        
        # Save lane to CSV
        csv_path = os.path.join(map_dir, f"{lane_name}.csv")
        save_csv(lane, csv_path, header=["#x_m", "y_m", "w_tr_right_m", "w_tr_left_m"])
        
        lanes.append(lane)
        lane_names.append(lane_name)
    
    return lanes, lane_names


def generate_raceline(lane_data, lane_name, args, module, map_dir):
    """Generate raceline trajectory for a given lane."""
    
    # Vehicle parameters
    veh_params = {
        "v_max": args.v_max,
        "length": args.vehicle_length,
        "width": args.vehicle_width,
        "mass": args.vehicle_mass,
        "dragcoeff": args.drag_coeff,
        "g": 9.81
    }

    # Calculation parameters
    stepsize_opts = {
        "stepsize_prep": 0.5,
        "stepsize_reg": 2.0,
        "stepsize_interp_after_opt": 0.2
    }

    # Smoothing parameters
    reg_smooth_opts = {
        "k_reg": 3,
        "s_reg": 0.0
    }

    # Velocity calculation options
    vel_calc_opts = {
        "dyn_model_exp": 1.0,
        "vel_profile_conv_filt_window": 31
    }

    # File paths
    file_paths = {
        "ggv_file": os.path.join("vehicle_dynamic_info", "ggv.csv"),
        "ax_max_machines_file": os.path.join("vehicle_dynamic_info", "ax_max_machines.csv")
    }

    # Import options
    imp_opts = {
        "flip_imp_track": False,
        "set_new_start": False, 
        "new_start": np.array([0.0, 0.0]),
        "min_track_width": veh_params["width"] * 2.0,
        "num_laps": args.num_laps
    }

    # Use lane data directly as reftrack
    reftrack_imp = lane_data

    # Import vehicle dynamics data
    ggv, ax_max_machines = tph.import_veh_dyn_info.import_veh_dyn_info(
        ggv_import_path=file_paths["ggv_file"],
        ax_max_machines_import_path=file_paths["ax_max_machines_file"]
    )

    # Adjust v_max if necessary
    max_ggv_velocity = np.max(ggv[:, 0])
    if veh_params["v_max"] > max_ggv_velocity:
        veh_params["v_max"] = max_ggv_velocity * 0.95

    # Smooth and interpolate
    reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp = \
        prep_track(
            reftrack_imp=reftrack_imp,
            reg_smooth_opts=reg_smooth_opts,
            stepsize_opts=stepsize_opts,
            debug=False,
            min_width=imp_opts["min_track_width"]
        )

    # Use lane directly (no optimization)
    alpha_opt = np.zeros(reftrack_interp.shape[0])

    # Create raceline trajectory
    raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp, s_points_opt_interp, \
        spline_lengths_opt, el_lengths_opt_interp = tph.create_raceline.create_raceline(
            refline=reftrack_interp[:, :2],
            normvectors=normvec_normalized_interp,
            alpha=alpha_opt,
            stepsize_interp=stepsize_opts["stepsize_interp_after_opt"]
        )

    # Calculate heading and curvature
    psi_vel_opt, kappa_opt = tph.calc_head_curv_an.calc_head_curv_an(
        coeffs_x=coeffs_x_opt,
        coeffs_y=coeffs_y_opt, 
        ind_spls=spline_inds_opt_interp,
        t_spls=t_vals_opt_interp
    )

    # Calculate velocity profile
    vx_profile_opt = tph.calc_vel_profile.calc_vel_profile(
        ggv=ggv,
        ax_max_machines=ax_max_machines,
        v_max=veh_params["v_max"],
        kappa=kappa_opt,
        el_lengths=el_lengths_opt_interp,
        closed=True,
        filt_window=vel_calc_opts["vel_profile_conv_filt_window"],
        dyn_model_exp=vel_calc_opts["dyn_model_exp"],
        drag_coeff=veh_params["dragcoeff"],
        m_veh=veh_params["mass"]
    )

    # Calculate acceleration profile
    vx_profile_opt_cl = np.append(vx_profile_opt, vx_profile_opt[0])
    ax_profile_opt = tph.calc_ax_profile.calc_ax_profile(
        vx_profile=vx_profile_opt_cl,
        el_lengths=el_lengths_opt_interp,
        eq_length_output=False
    )

    # Calculate lap time
    t_profile_cl = tph.calc_t_profile.calc_t_profile(
        vx_profile=vx_profile_opt,
        ax_profile=ax_profile_opt,
        el_lengths=el_lengths_opt_interp
    )

    # Assemble final trajectory
    trajectory = np.column_stack((
        s_points_opt_interp,
        raceline_interp,
        psi_vel_opt + 0.5 * np.pi,
        kappa_opt,
        vx_profile_opt,
        ax_profile_opt
    ))

    # Create closed trajectory
    traj_cl = np.vstack((trajectory, trajectory[0, :]))
    traj_cl[-1, 0] = np.sum(spline_lengths_opt)

    return traj_cl, t_profile_cl[-1]

def main():
    args = parse_arguments()
    module = os.path.dirname(os.path.abspath(__file__))
    
    # Define base directory and map paths
    map_dir = os.path.join(args.map_name)
    
    # Create output directory
    os.makedirs(map_dir, exist_ok=True)
    
    print(f"Generating lanes for {args.map_name}...")
    # Generate lanes
    lanes, lane_names = generate_lanes(args, map_dir)
    
    # Generate raceline for each lane
    for idx, (lane, lane_name) in enumerate(zip(lanes, lane_names)):
        print(f"\nProcessing {lane_name}...")
        
        # Generate raceline trajectory
        trajectory, laptime = generate_raceline(lane, lane_name, args, module, map_dir)
        # Use unified naming: raceline0, raceline1, raceline2, etc.
        export_path = os.path.join(map_dir, f"raceline{idx}.csv")
        
        header = "s_m;x_m;y_m;psi_rad;kappa_radpm;vx_mps;ax_mps2"
        np.savetxt(export_path, trajectory, delimiter=";", fmt="%.6f", header=header)
        
        print(f"  Estimated laptime: {laptime:.2f}s")
        print(f"  Trajectory exported to: {export_path}")


if __name__ == "__main__":
    main()