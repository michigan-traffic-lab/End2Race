# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Rendering engine for f1tenth gym env based on pyglet and OpenGL
Author: Hongrui Zheng
"""

# opengl stuff
import pyglet
from pyglet.gl import *

# other
import cv2
import numpy as np
from PIL import Image
import yaml

# helpers
from f110_gym.envs.collision_models import get_vertices

# zooming constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1 / ZOOM_IN_FACTOR

# vehicle shape constants
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31

BACKGROUND_COLOR = (244, 241, 234)
TRACK_BOUNDARY_COLOR = (31, 35, 38)
EGO_COLOR = (79, 146, 196)
OPPONENT_COLOR = (214, 111, 102)
VEHICLE_OUTLINE_COLOR = (35, 39, 42)


class EnvRenderer(pyglet.window.Window):
    """
    A window class inherited from pyglet.window.Window, handles the camera/projection interaction, resizing window, and rendering the environment
    """

    def __init__(self, width, height, *args, **kwargs):
        """
        Class constructor

        Args:
            width (int): width of the window
            height (int): height of the window

        Returns:
            None
        """
        conf = Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True)
        super().__init__(
            width, height, config=conf, resizable=False, vsync=False, *args, **kwargs
        )

        # gl init
        glClearColor(*(channel / 255 for channel in BACKGROUND_COLOR), 1.0)
        glPointSize(4.0)
        glLineWidth(3.5)

        # initialize camera values
        self.left = -width / 2
        self.right = width / 2
        self.bottom = -height / 2
        self.top = height / 2
        self.zoom_level = 1.2
        self.zoomed_width = width
        self.zoomed_height = height

        # current batch that keeps track of all graphics
        self.batch = pyglet.graphics.Batch()

        # current env map
        self.map_points = None

        # current env agent poses, (num_agents, 3), columns are (x, y, theta)
        self.poses = None

        # current env agent vertices, (num_agents, 4, 2), 2nd and 3rd dimensions are the 4 corners in 2D
        self.vertices = None

        # current score label
        self.score_label = pyglet.text.Label(
            "Lap Time: {laptime:.2f}, Ego Lap Count: {count:.0f}".format(
                laptime=0.0, count=0.0
            ),
            font_name="Arial",
            font_size=6,
            x=0,
            y=-800,
            anchor_x="left",
            anchor_y="top",
            # width=0.01,
            # height=0.01,
            color=(*VEHICLE_OUTLINE_COLOR, 255),
            batch=self.batch,
        )

        self.fps_display = pyglet.window.FPSDisplay(self)

    def update_map(self, map_path, map_ext):
        """
        Update the map being drawn by the renderer. Converts image to a list of 3D points representing each obstacle pixel in the map.

        Args:
            map_path (str): absolute path to the map without extensions
            map_ext (str): extension for the map image file

        Returns:
            None
        """

        # load map metadata
        with open(map_path + ".yaml", "r") as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                map_resolution = map_metadata["resolution"]
                origin = map_metadata["origin"]
                origin_x = origin[0]
                origin_y = origin[1]
            except yaml.YAMLError as ex:
                print(ex)

        # load map image
        map_img = np.array(
            Image.open(map_path + map_ext).transpose(Image.FLIP_TOP_BOTTOM)
        ).astype(np.float64)
        map_height = map_img.shape[0]
        map_width = map_img.shape[1]

        occupied = (map_img <= 128.0).astype(np.uint8) * 255
        centerlines = cv2.ximgproc.thinning(occupied)
        component_count, labels = cv2.connectedComponents(centerlines)
        map_segments = []
        for component in range(1, component_count):
            component_mask = (labels == component).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                continue
            pixels = contours[0][:, 0, :].astype(np.float64)
            pixels = sum(np.roll(pixels, shift, axis=0) for shift in range(-3, 4)) / 7.0
            points = np.column_stack(
                (
                    50.0 * (pixels[:, 0] * map_resolution + origin_x),
                    50.0 * (pixels[:, 1] * map_resolution + origin_y),
                    np.zeros(len(pixels)),
                )
            )
            map_segments.append(
                np.column_stack((points, np.roll(points, -1, axis=0))).reshape(-1, 3)
            )
        map_vertices = np.concatenate(map_segments)
        self.batch.add(
            len(map_vertices),
            GL_LINES,
            None,
            ("v3f/static", map_vertices.flatten().tolist()),
            ("c3B/static", list(TRACK_BOUNDARY_COLOR) * len(map_vertices)),
        )
        self.map_points = map_vertices

    def on_resize(self, width, height):
        """
        Callback function on window resize, overrides inherited method, and updates camera values on top of the inherited on_resize() method.

        Potential improvements on current behavior: zoom/pan resets on window resize.

        Args:
            width (int): new width of window
            height (int): new height of window

        Returns:
            None
        """

        # call overrided function
        super().on_resize(width, height)

        # update camera value
        (width, height) = self.get_size()
        self.left = -self.zoom_level * width / 2
        self.right = self.zoom_level * width / 2
        self.bottom = -self.zoom_level * height / 2
        self.top = self.zoom_level * height / 2
        self.zoomed_width = self.zoom_level * width
        self.zoomed_height = self.zoom_level * height

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """
        Callback function on mouse drag, overrides inherited method.

        Args:
            x (int): Distance in pixels from the left edge of the window.
            y (int): Distance in pixels from the bottom edge of the window.
            dx (int): Relative X position from the previous mouse position.
            dy (int): Relative Y position from the previous mouse position.
            buttons (int): Bitwise combination of the mouse buttons currently pressed.
            modifiers (int): Bitwise combination of any keyboard modifiers currently active.

        Returns:
            None
        """

        # pan camera
        self.left -= dx * self.zoom_level
        self.right -= dx * self.zoom_level
        self.bottom -= dy * self.zoom_level
        self.top -= dy * self.zoom_level

    def on_mouse_scroll(self, x, y, dx, dy):
        """
        Callback function on mouse scroll, overrides inherited method.

        Args:
            x (int): Distance in pixels from the left edge of the window.
            y (int): Distance in pixels from the bottom edge of the window.
            scroll_x (float): Amount of movement on the horizontal axis.
            scroll_y (float): Amount of movement on the vertical axis.

        Returns:
            None
        """

        # Get scale factor
        f = ZOOM_IN_FACTOR if dy > 0 else ZOOM_OUT_FACTOR if dy < 0 else 1

        # If zoom_level is in the proper range
        if 0.01 < self.zoom_level * f < 10:

            self.zoom_level *= f

            (width, height) = self.get_size()

            mouse_x = x / width
            mouse_y = y / height

            mouse_x_in_world = self.left + mouse_x * self.zoomed_width
            mouse_y_in_world = self.bottom + mouse_y * self.zoomed_height

            self.zoomed_width *= f
            self.zoomed_height *= f

            self.left = mouse_x_in_world - mouse_x * self.zoomed_width
            self.right = mouse_x_in_world + (1 - mouse_x) * self.zoomed_width
            self.bottom = mouse_y_in_world - mouse_y * self.zoomed_height
            self.top = mouse_y_in_world + (1 - mouse_y) * self.zoomed_height

    def on_close(self):
        """
        Callback function when the 'x' is clicked on the window, overrides inherited method. Also throws exception to end the python program when in a loop.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: with a message that indicates the rendering window was closed
        """

        super().on_close()
        raise Exception("Rendering window was closed.")

    def on_draw(self):
        """
        Function when the pyglet is drawing. The function draws the batch created that includes the map points, the agent polygons, and the information text, and the fps display.

        Args:
            None

        Returns:
            None
        """

        # if map and poses doesn't exist, raise exception
        if self.map_points is None:
            raise Exception("Map not set for renderer.")
        if self.poses is None:
            raise Exception("Agent poses not updated for renderer.")

        # Initialize Projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # Initialize Modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Save the default modelview matrix
        glPushMatrix()

        # Clear window with ClearColor
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set orthographic projection matrix
        glOrtho(self.left, self.right, self.bottom, self.top, 1, -1)

        # Draw all batches
        self.batch.draw()
        # Remove default modelview matrix
        glPopMatrix()

    def update_obs(self, obs):
        """
        Updates the renderer with the latest observation from the gym environment, including the agent poses, and the information text.

        Args:
            obs (dict): observation dict from the gym env

        Returns:
            None
        """

        self.ego_idx = obs["ego_idx"]
        poses_x = obs["poses_x"]
        poses_y = obs["poses_y"]
        poses_theta = obs["poses_theta"]

        num_agents = len(poses_x)
        if self.poses is None:
            self.cars = []
            self.car_outlines = []
            for i in range(num_agents):
                vertices_np = get_vertices(
                    np.array([0.0, 0.0, 0.0]), CAR_LENGTH, CAR_WIDTH
                )
                vertices = list(vertices_np.flatten())
                body_color = EGO_COLOR if i == self.ego_idx else OPPONENT_COLOR
                self.car_outlines.append(
                    self.batch.add(
                        4,
                        GL_QUADS,
                        None,
                        ("v2f/stream", vertices),
                        ("c3B/static", list(VEHICLE_OUTLINE_COLOR) * 4),
                    )
                )
                self.cars.append(
                    self.batch.add(
                        4,
                        GL_QUADS,
                        None,
                        ("v2f/stream", vertices),
                        ("c3B/static", list(body_color) * 4),
                    )
                )

        poses = np.stack((poses_x, poses_y, poses_theta)).T
        for j in range(poses.shape[0]):
            vertices_np = 50.0 * get_vertices(poses[j, :], CAR_LENGTH, CAR_WIDTH)
            vertices = list(vertices_np.flatten())
            self.cars[j].vertices = vertices
            outline_vertices = 50.0 * get_vertices(
                poses[j, :], CAR_LENGTH + 0.05, CAR_WIDTH + 0.05
            )
            self.car_outlines[j].vertices = outline_vertices.flatten().tolist()
        self.poses = poses

        self.score_label.text = (
            "Lap Time: {laptime:.2f}, Ego Lap Count: {count:.0f}".format(
                laptime=obs["lap_times"][0], count=obs["lap_counts"][obs["ego_idx"]]
            )
        )
