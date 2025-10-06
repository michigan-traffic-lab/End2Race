from latticeplanner.utils import *

import numpy as np

class PurePursuitPlanner:
    def __init__(self, conf, wpt_path, wb=0.33):
        """
        conf: NameSpace
        """

        self.wheelbase = wb
        self.conf = conf
        self.max_reacquire = 20.0
        self.wpt_path = wpt_path

        self.drawn_waypoints = []
        # waypoint index
        self.wpt_xind = self.conf.wpt_xind
        self.wpt_yind = self.conf.wpt_yind
        self.wpt_vind = self.conf.wpt_vind
        self.waypoints_xyv = None
        self.waypoints = None
        self.load_waypoints()
        self.wpNum = self.waypoints.shape[0]

        # advanced pure pursuit
        self.minL = self.conf.minL
        self.maxL = self.conf.maxL
        self.Lscale = self.conf.Lscale
        # steering gain tuning
        self.minP = self.conf.minP
        self.maxP = self.conf.maxP
        self.Pscale = self.conf.Pscale
        # D-term for damping
        self.D = self.conf.D
        self.prev_error = 0.0
        # speed scaling & interpolation
        self.vel_scale = self.conf.vel_scale
        self.interpScale = self.conf.interpScale

    def _change_waypoint_xyv_idx(self, new_x_idx, new_y_idx, new_v_idx):
        self.wpt_xind = new_x_idx
        self.wpt_yind = new_y_idx
        self.wpt_vind = new_v_idx
        print('change waypoint x, y, v idx')

    def load_waypoints(self):
        """
        loads waypoints
        """
        waypoints = np.loadtxt(self.wpt_path, delimiter=';', skiprows=2)
        waypoints = np.vstack((waypoints[:, 1], waypoints[:, 2], waypoints[:, 5], waypoints[:, 3], waypoints[:, 0])).T
        self.waypoints = waypoints
        self.waypoints_xyv = waypoints[:, :3]

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """

        # points = self.waypoints
        from pyglet.gl import GL_POINTS
        points = np.vstack((self.waypoints[:, 0], self.waypoints[:, 1])).T

        scaled_points = 50. * points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    def _get_current_waypoint(self, lookahead_distance, position, theta, waypoints):
        """
        gets the current waypoint to follow
        """
        nearest_p, nearest_dist, t, i = nearest_point(position, self.waypoints[:, 0:2])
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = intersect_point(position,
                                                      lookahead_distance,
                                                      self.waypoints[:, 0:2],
                                                      i + t,
                                                      wrap=True)
            if i2 is None:
                all_distance = np.linalg.norm(self.waypoints[:, 0:2] - position, axis=1)
                all_distance_lh = np.abs(all_distance - lookahead_distance)
                best_p_idx = np.argmin(all_distance_lh)
                return self.waypoints[best_p_idx, :]
            current_waypoint = np.array([self.waypoints[i2, 0], self.waypoints[i2, 1], self.waypoints[i, 2]])
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return self.waypoints[i, :]
        else:
            return None

    def get_L(self, curr_v):
        return curr_v * (self.maxL - self.minL) / self.Lscale + self.minL

    def plan(self, pose_x, pose_y, pose_theta, curr_v, waypoints):

        # get L, P with speed
        L = curr_v * (self.maxL - self.minL) / self.Lscale + self.minL
        P = self.maxP - curr_v * (self.maxP - self.minP) / self.Pscale

        position = np.array([pose_x, pose_y])
        lookahead_point, new_L, nearest_dist = get_wp_xyv_with_interp(L, position, pose_theta, waypoints, waypoints.shape[0], self.interpScale)
        self.nearest_dist = nearest_dist

        speed, steering, error = \
            get_actuation_PD(pose_theta, lookahead_point, position, new_L, self.wheelbase, self.prev_error, P, self.D)
        speed = speed * self.vel_scale
        self.prev_error = error

        return steering, speed


@njit(cache=True)
def simple_norm_axis1(vector):
    return np.sqrt(vector[:, 0]**2 + vector[:, 1]**2)


@njit(cache=True)
def get_wp_xyv_with_interp(L, curr_pos, theta, waypoints, wpNum, interpScale):
    traj_distances = simple_norm_axis1(waypoints[:, :2] - curr_pos)
    nearest_idx = np.argmin(traj_distances)
    nearest_dist = traj_distances[nearest_idx]
    segment_end = nearest_idx

    if traj_distances[-1] < L:
        segment_end = wpNum-1

    else:
        while traj_distances[segment_end] < L:
            segment_end = (segment_end + 1) % wpNum

    segment_begin = (segment_end - 1 + wpNum) % wpNum
    x_array = np.linspace(waypoints[segment_begin, 0], waypoints[segment_end, 0], interpScale)
    y_array = np.linspace(waypoints[segment_begin, 1], waypoints[segment_end, 1], interpScale)
    v_array = np.linspace(waypoints[segment_begin, 2], waypoints[segment_end, 2], interpScale)
    xy_interp = np.vstack((x_array, y_array)).T
    dist_interp = simple_norm_axis1(xy_interp - curr_pos) - L
    i_interp = np.argmin(np.abs(dist_interp))
    target_global = np.array((x_array[i_interp], y_array[i_interp]))
    new_L = np.linalg.norm(curr_pos - target_global)
    return np.array((x_array[i_interp], y_array[i_interp], v_array[i_interp])), new_L, nearest_dist
