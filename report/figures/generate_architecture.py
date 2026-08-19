from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch


ROOT = Path(__file__).resolve().parents[2]
TRACK_ROOT = ROOT / "f1tenth_sim" / "f1tenth_racetracks"
OUTPUT_DIR = Path(__file__).resolve().parent
PLATFORM_IMAGE_PATH = OUTPUT_DIR / "f1tenth-platform.png"

NAVY = "#17324D"
TEAL = "#2B8C83"
BLUE = "#3E73A8"
CORAL = "#C9675D"
GOLD = "#C99A3D"
INK = "#243746"
MID = "#758590"
LIGHT = "#D9E1E5"
PANEL = "#F7F9FA"
WHITE = "#FFFFFF"
PALE_TEAL = "#E9F3F1"
PALE_BLUE = "#EBF1F7"
PALE_CORAL = "#F6ECEA"
PALE_GOLD = "#F6F0E3"


def panel_box(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.005, 0.005),
            0.99,
            0.99,
            boxstyle="round,pad=0.006,rounding_size=0.018",
            facecolor=PANEL,
            edgecolor=LIGHT,
            linewidth=0.8,
            zorder=0,
        )
    )


def panel_header(ax, number, title, color):
    ax.add_patch(Rectangle((0.03, 0.91), 0.94, 0.055, facecolor=color, alpha=0.10, edgecolor="none"))
    ax.text(0.055, 0.938, number, color=color, fontsize=10, fontweight="bold", va="center")
    ax.text(0.16, 0.938, title, color=NAVY, fontsize=9.3, fontweight="bold", va="center")


def label(
    ax,
    x,
    y,
    text,
    color=INK,
    size=6.5,
    weight="normal",
    ha="center",
    va="center",
    rotation=0,
):
    ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=size,
        fontweight=weight,
        ha=ha,
        va=va,
        rotation=rotation,
    )


def chip(ax, x, y, width, height, text, face=WHITE, edge=LIGHT, color=INK, size=6.2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.8,
        )
    )
    label(ax, x + width / 2, y + height / 2, text, color=color, size=size, weight="bold")


def arrow(ax, start, end, color=NAVY, width=0.8, style="-|>", connectionstyle="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=7,
            linewidth=width,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=1,
            shrinkB=1,
        )
    )


def rotated(points, x, y, angle, scale=1.0):
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.asarray(points) * scale @ rotation.T + np.array([x, y])


def draw_car(ax, x, y, length, width, angle=0, color=BLUE, lidar=False, zorder=5):
    body = np.array([
        [-0.48, -0.36],
        [0.28, -0.36],
        [0.50, -0.22],
        [0.50, 0.22],
        [0.28, 0.36],
        [-0.48, 0.36],
    ])
    body[:, 0] *= length
    body[:, 1] *= width
    ax.add_patch(
        Polygon(rotated(body, x, y, angle), closed=True, facecolor=color, edgecolor=NAVY, linewidth=0.7, zorder=zorder)
    )
    cabin = np.array([[-0.08, -0.24], [0.25, -0.24], [0.34, 0.0], [0.25, 0.24], [-0.08, 0.24]])
    cabin[:, 0] *= length
    cabin[:, 1] *= width
    ax.add_patch(
        Polygon(rotated(cabin, x, y, angle), closed=True, facecolor=WHITE, edgecolor=NAVY, linewidth=0.5, zorder=zorder + 1)
    )
    for dx in (-0.29, 0.29):
        for dy in (-0.43, 0.43):
            wheel = np.array([[-0.09, -0.07], [0.09, -0.07], [0.09, 0.07], [-0.09, 0.07]])
            wheel[:, 0] *= length
            wheel[:, 1] *= width
            center = rotated([[dx * length, dy * width]], x, y, angle)[0]
            ax.add_patch(
                Polygon(rotated(wheel, center[0], center[1], angle), closed=True, facecolor=INK, edgecolor="none", zorder=zorder - 1)
            )
    if lidar:
        ax.add_patch(Circle((x, y), 0.065 * length, facecolor=TEAL, edgecolor=NAVY, linewidth=0.5, zorder=zorder + 2))


def plot_track(ax, map_name, color, tag, tag_color):
    path = TRACK_ROOT / map_name / "raceline1.csv"
    values = np.loadtxt(path, delimiter=";", skiprows=1, ndmin=2)
    points = values[:, 1:3]
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.2, solid_capstyle="round")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.08)
    ax.axis("off")
    display_name = "MOSCOW RACEWAY" if map_name == "MoscowRaceway" else map_name.upper()
    ax.text(0.5, -0.02, display_name, transform=ax.transAxes, ha="center", va="top", fontsize=4.9, color=INK, fontweight="bold")
    ax.text(0.5, -0.16, tag, transform=ax.transAxes, ha="center", va="top", fontsize=4.6, color=tag_color, fontweight="bold")


def bezier(start, control_1, control_2, end, samples=80):
    t = np.linspace(0.0, 1.0, samples)[:, None]
    return (
        (1 - t) ** 3 * np.asarray(start)
        + 3 * (1 - t) ** 2 * t * np.asarray(control_1)
        + 3 * (1 - t) * t**2 * np.asarray(control_2)
        + t**3 * np.asarray(end)
    )


def draw_lidar_icon(ax, x, y, radius, beams=24):
    for index, theta in enumerate(np.linspace(0, 2 * np.pi, beams, endpoint=False)):
        reach = radius * (0.72 + 0.20 * np.sin(index * 1.7) ** 2)
        ax.plot(
            [x + 0.18 * radius * np.cos(theta), x + reach * np.cos(theta)],
            [y + 0.18 * radius * np.sin(theta), y + reach * np.sin(theta)],
            color=TEAL,
            linewidth=0.45,
            alpha=0.85,
        )
    ax.add_patch(Circle((x, y), radius, fill=False, edgecolor=LIGHT, linewidth=0.7))
    ax.add_patch(Circle((x, y), 0.11 * radius, facecolor=NAVY, edgecolor="none"))


def racing_scenarios_panel(ax):
    panel_box(ax)
    panel_header(ax, "01", "RACING SCENARIOS", TEAL)

    tracks = [
        ("Austin", TEAL, "TRAIN", TEAL),
        ("Hockenheim", TEAL, "TEST", MID),
        ("MoscowRaceway", TEAL, "TEST", MID),
        ("Nuerburgring", TEAL, "TEST", MID),
    ]
    positions = [(0.055, 0.715), (0.535, 0.715), (0.055, 0.555), (0.535, 0.555)]
    for position, track in zip(positions, tracks):
        inset = ax.inset_axes([position[0], position[1], 0.41, 0.12])
        plot_track(inset, *track)

    label(ax, 0.25, 0.505, "F1TENTH PLATFORM", color=NAVY, size=5.7, weight="bold")
    ax.add_patch(
        FancyBboxPatch(
            (0.05, 0.282),
            0.40,
            0.195,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            facecolor=WHITE,
            edgecolor=LIGHT,
            linewidth=0.7,
        )
    )
    platform_ax = ax.inset_axes([0.06, 0.292, 0.38, 0.175])
    platform_image = mpimg.imread(PLATFORM_IMAGE_PATH)
    platform_ax.imshow(platform_image[300:1589, 0:1660], interpolation="lanczos")
    platform_ax.axis("off")

    label(ax, 0.74, 0.505, "F1TENTH SIM", color=NAVY, size=5.7, weight="bold")
    ax.add_patch(
        FancyBboxPatch(
            (0.54, 0.282),
            0.40,
            0.195,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            facecolor=WHITE,
            edgecolor=LIGHT,
            linewidth=0.7,
        )
    )
    simulated_path = bezier((0.57, 0.31), (0.64, 0.39), (0.78, 0.39), (0.91, 0.45))
    ax.plot(simulated_path[:, 0], simulated_path[:, 1], color=LIGHT, linewidth=10, solid_capstyle="round")
    ax.plot(simulated_path[:, 0], simulated_path[:, 1], color=WHITE, linewidth=0.7, linestyle=(0, (3, 3)))
    draw_car(ax, 0.68, 0.365, 0.055, 0.060, angle=28, color=BLUE)
    draw_car(ax, 0.79, 0.398, 0.055, 0.060, angle=28, color=CORAL)
    for theta in np.linspace(-0.6, 0.6, 7):
        ax.plot(
            [0.68, 0.68 + 0.085 * np.cos(theta)],
            [0.365, 0.365 + 0.060 * np.sin(theta)],
            color=TEAL,
            linewidth=0.4,
            alpha=0.7,
        )
    label(ax, 0.74, 0.255, "2D multi-agent simulation", color=MID, size=4.8)

    ax.plot([0.06, 0.94], [0.225, 0.225], color=LIGHT, linewidth=0.7)
    label(ax, 0.50, 0.195, "SCENARIO GENERATOR", color=NAVY, size=5.8, weight="bold")
    chip(ax, 0.07, 0.115, 0.24, 0.055, "80 STARTS", face=WHITE, edge=LIGHT, size=5.3)
    chip(ax, 0.38, 0.115, 0.24, 0.055, "3 RACELINES", face=WHITE, edge=LIGHT, size=5.2)
    chip(ax, 0.69, 0.115, 0.24, 0.055, "3 SPEEDS", face=WHITE, edge=LIGHT, size=5.3)
    label(ax, 0.50, 0.087, "80 × 3 × 3", color=MID, size=5.0)
    arrow(ax, (0.50, 0.078), (0.50, 0.064), color=TEAL)
    chip(ax, 0.34, 0.018, 0.32, 0.050, "720 SCENARIOS", face=PALE_TEAL, edge=TEAL, color=TEAL, size=6.0)


def expert_panel(ax):
    panel_box(ax)
    panel_header(ax, "02", "EXPERT DEMONSTRATIONS", TEAL)
    label(ax, 0.06, 0.872, "LATTICE PLANNER", color=NAVY, size=6.7, weight="bold", ha="left")

    road_y = np.linspace(0.47, 0.83, 120)
    center = 0.48 + 0.11 * np.sin((road_y - 0.47) * 7.2)
    ax.fill_betweenx(road_y, center - 0.30, center + 0.30, color="#EEF1F2", zorder=0)
    ax.plot(center - 0.30, road_y, color=MID, linewidth=0.9)
    ax.plot(center + 0.30, road_y, color=MID, linewidth=0.9)
    ax.plot(center, road_y, color=WHITE, linewidth=0.8, linestyle=(0, (4, 4)))

    start = (0.43, 0.49)
    terminal_x = [0.25, 0.36, 0.48, 0.60, 0.71]
    for index, end_x in enumerate(terminal_x):
        path = bezier(start, (0.43, 0.60), (end_x, 0.70), (end_x, 0.81))
        selected = index == 1
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=TEAL if selected else MID,
            linewidth=1.7 if selected else 0.7,
            linestyle="-" if selected else (0, (3, 3)),
            alpha=1.0 if selected else 0.65,
            zorder=2,
        )
    draw_car(ax, 0.43, 0.505, 0.10, 0.10, angle=90, color=BLUE)
    draw_car(ax, 0.39, 0.718, 0.095, 0.10, angle=82, color=CORAL)
    obstacle_angles = np.linspace(-0.7, 0.7, 11)
    for theta in obstacle_angles:
        ax.add_patch(Circle((0.39 + 0.075 * np.cos(theta), 0.718 + 0.055 * np.sin(theta)), 0.006, facecolor=CORAL, edgecolor="none", zorder=4))
    label(ax, 0.57, 0.742, "LiDAR occupancy", color=CORAL, size=5.0, ha="left")
    label(ax, 0.25, 0.826, "selected trajectory", color=TEAL, size=5.1)

    ax.plot([0.06, 0.94], [0.435, 0.435], color=LIGHT, linewidth=0.7)
    label(ax, 0.06, 0.400, "PURE PURSUIT TRACKING", color=NAVY, size=6.7, weight="bold", ha="left")
    tracking_path = bezier((0.14, 0.15), (0.17, 0.29), (0.36, 0.33), (0.48, 0.25))
    ax.plot(tracking_path[:, 0], tracking_path[:, 1], color=TEAL, linewidth=1.4)
    draw_car(ax, 0.15, 0.16, 0.10, 0.10, angle=78, color=BLUE)
    target = tracking_path[58]
    ax.add_patch(Circle(target, 0.017, facecolor=WHITE, edgecolor=TEAL, linewidth=1.0))
    ax.plot([0.15, target[0]], [0.16, target[1]], color=MID, linewidth=0.7, linestyle=(0, (2, 2)))
    label(ax, target[0] + 0.01, target[1] + 0.035, "lookahead", color=TEAL, size=5.0)
    arrow(ax, (0.50, 0.245), (0.57, 0.245), color=TEAL, width=1.0)
    chip(ax, 0.60, 0.275, 0.29, 0.065, r"$\delta$  STEERING", face=WHITE, edge=BLUE, color=NAVY, size=5.7)
    chip(ax, 0.60, 0.175, 0.29, 0.065, r"$v$  DESIRED SPEED", face=WHITE, edge=TEAL, color=NAVY, size=5.7)
    chip(ax, 0.12, 0.055, 0.33, 0.055, "10 Hz PLANNING", face=PALE_TEAL, edge=TEAL, color=TEAL, size=5.5)
    chip(ax, 0.55, 0.055, 0.33, 0.055, "40 Hz TRACKING", face=PALE_BLUE, edge=BLUE, color=BLUE, size=5.5)


def imitation_panel(ax):
    panel_box(ax)
    panel_header(ax, "03", "IMITATION LEARNING", BLUE)
    label(ax, 0.08, 0.867, "POLICY INPUTS", color=NAVY, size=6.7, weight="bold", ha="left")

    draw_lidar_icon(ax, 0.19, 0.765, 0.075, beams=20)
    label(ax, 0.19, 0.673, "180 LiDAR", color=INK, size=5.5, weight="bold")
    ax.add_patch(Circle((0.53, 0.765), 0.068, facecolor=WHITE, edgecolor=LIGHT, linewidth=0.8))
    label(ax, 0.53, 0.773, r"$v_{t-1}$", color=BLUE, size=8.0, weight="bold")
    label(ax, 0.53, 0.673, "PREVIOUS SPEED", color=INK, size=5.5, weight="bold")

    chip(ax, 0.09, 0.565, 0.22, 0.060, "SIGMOID", face=PALE_TEAL, edge=TEAL, color=TEAL, size=5.8)
    chip(ax, 0.42, 0.565, 0.22, 0.060, "LINEAR 1→30", face=PALE_BLUE, edge=BLUE, color=BLUE, size=5.6)
    arrow(ax, (0.19, 0.69), (0.19, 0.63), color=TEAL)
    arrow(ax, (0.53, 0.69), (0.53, 0.63), color=BLUE)
    label(ax, 0.20, 0.535, "pressure tokens", color=TEAL, size=4.9)
    label(ax, 0.53, 0.535, "speed embedding", color=BLUE, size=4.9)

    ax.add_patch(Circle((0.37, 0.485), 0.025, facecolor=WHITE, edgecolor=NAVY, linewidth=0.9))
    label(ax, 0.37, 0.486, "+", color=NAVY, size=7.5, weight="bold")
    arrow(ax, (0.20, 0.555), (0.35, 0.505), color=TEAL)
    arrow(ax, (0.53, 0.555), (0.39, 0.505), color=BLUE)

    chip(ax, 0.22, 0.375, 0.30, 0.075, "GRU  ·  420", face=WHITE, edge=BLUE, color=NAVY, size=7.0)
    arrow(ax, (0.37, 0.46), (0.37, 0.45), color=NAVY)
    arrow(ax, (0.52, 0.412), (0.62, 0.412), color=BLUE, connectionstyle="arc3,rad=-0.8")
    arrow(ax, (0.62, 0.412), (0.52, 0.412), color=BLUE, connectionstyle="arc3,rad=-0.8")
    label(ax, 0.61, 0.474, "hidden state", color=BLUE, size=4.8)
    chip(ax, 0.22, 0.265, 0.30, 0.065, "MLP  ·  420→128→2", face=PALE_GOLD, edge=GOLD, color=NAVY, size=5.8)
    arrow(ax, (0.37, 0.375), (0.37, 0.335), color=NAVY)
    chip(ax, 0.08, 0.145, 0.27, 0.065, r"$\delta$  STEERING", face=WHITE, edge=BLUE, color=NAVY, size=5.6)
    chip(ax, 0.40, 0.145, 0.27, 0.065, r"$v$  DESIRED SPEED", face=WHITE, edge=TEAL, color=NAVY, size=5.5)
    arrow(ax, (0.34, 0.265), (0.22, 0.215), color=NAVY)
    arrow(ax, (0.40, 0.265), (0.53, 0.215), color=NAVY)

    ax.plot([0.73, 0.73], [0.11, 0.84], color=LIGHT, linewidth=0.7)
    label(ax, 0.83, 0.795, "BEHAVIORAL", color=NAVY, size=5.5, weight="bold")
    label(ax, 0.83, 0.765, "CLONING", color=NAVY, size=5.5, weight="bold")
    for row, row_color in enumerate((TEAL, BLUE, TEAL, BLUE)):
        y = 0.665 - 0.06 * row
        ax.plot([0.78, 0.88], [y, y], color=row_color, linewidth=1.1)
        ax.add_patch(Circle((0.78, y), 0.007, facecolor=row_color, edgecolor="none"))
        ax.add_patch(Circle((0.88, y), 0.007, facecolor=row_color, edgecolor="none"))
    label(ax, 0.83, 0.415, "expert actions", color=MID, size=4.9)
    chip(ax, 0.77, 0.285, 0.13, 0.07, "MSE", face=PALE_CORAL, edge=CORAL, color=CORAL, size=6.0)
    arrow(ax, (0.83, 0.40), (0.83, 0.36), color=CORAL)
    arrow(ax, (0.67, 0.178), (0.77, 0.305), color=MID, connectionstyle="arc3,rad=-0.2")
    label(ax, 0.83, 0.225, r"$\mathcal{L}_{steer}+0.05\mathcal{L}_{speed}$", color=INK, size=5.2)
    label(ax, 0.83, 0.120, "500 epochs", color=MID, size=5.0)


def simulator_icon(ax, x, y, width, height):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.005,rounding_size=0.01", facecolor=WHITE, edgecolor=LIGHT, linewidth=0.7))
    ax.plot([x + 0.03, x + width - 0.03], [y + height * 0.30, y + height * 0.70], color=MID, linewidth=6, solid_capstyle="round")
    ax.plot([x + 0.03, x + width - 0.03], [y + height * 0.30, y + height * 0.70], color=WHITE, linewidth=0.6, linestyle=(0, (3, 3)))
    draw_car(ax, x + width * 0.43, y + height * 0.46, width * 0.13, height * 0.17, angle=25, color=BLUE)
    draw_car(ax, x + width * 0.63, y + height * 0.58, width * 0.13, height * 0.17, angle=25, color=CORAL)


def reinforcement_panel(ax):
    panel_box(ax)
    panel_header(ax, "04", "PPO FINE-TUNING", GOLD)
    label(ax, 0.08, 0.867, "CLOSED-LOOP TRAINING", color=NAVY, size=6.7, weight="bold", ha="left")

    chip(ax, 0.08, 0.665, 0.29, 0.095, "RECURRENT\nPOLICY", face=PALE_BLUE, edge=BLUE, color=NAVY, size=6.2)
    simulator_icon(ax, 0.57, 0.655, 0.32, 0.12)
    label(ax, 0.73, 0.795, "RACING SIMULATOR", color=NAVY, size=5.7, weight="bold")
    arrow(ax, (0.38, 0.713), (0.56, 0.713), color=NAVY)
    label(ax, 0.47, 0.740, "action", color=MID, size=4.8)
    arrow(ax, (0.73, 0.65), (0.73, 0.565), color=NAVY)
    label(ax, 0.78, 0.605, "transition", color=MID, size=4.8, ha="left")

    chip(ax, 0.49, 0.455, 0.47, 0.095, r"$r_t=0.02\,\Delta s_t-\mathbf{1}_{collision}$", face=PALE_CORAL, edge=CORAL, color=INK, size=6.4)
    label(ax, 0.73, 0.425, "progress reward + collision penalty", color=MID, size=4.8)
    arrow(ax, (0.49, 0.505), (0.35, 0.505), color=CORAL)
    chip(ax, 0.08, 0.455, 0.27, 0.095, "GAE", face=PALE_GOLD, edge=GOLD, color=NAVY, size=7.0)
    arrow(ax, (0.215, 0.455), (0.215, 0.365), color=GOLD)
    chip(ax, 0.08, 0.265, 0.29, 0.095, "PPO-CLIP\nUPDATE", face=WHITE, edge=GOLD, color=NAVY, size=6.2)
    arrow(ax, (0.08, 0.31), (0.045, 0.31), color=GOLD, connectionstyle="arc3,rad=-0.5")
    arrow(ax, (0.045, 0.31), (0.14, 0.66), color=GOLD, connectionstyle="arc3,rad=-0.15")
    label(ax, 0.055, 0.52, "updated weights", color=GOLD, size=4.7, rotation=90)

    ax.plot([0.45, 0.45], [0.08, 0.38], color=LIGHT, linewidth=0.7)
    label(ax, 0.69, 0.348, "DETERMINISTIC SCREENING", color=NAVY, size=6.1, weight="bold")
    chip(ax, 0.52, 0.245, 0.35, 0.065, "720 SCENARIOS", face=WHITE, edge=LIGHT, color=INK, size=5.6)
    chip(ax, 0.52, 0.155, 0.16, 0.06, "SAFETY", face=PALE_TEAL, edge=TEAL, color=TEAL, size=5.3)
    chip(ax, 0.71, 0.155, 0.16, 0.06, "OVERTAKE", face=PALE_BLUE, edge=BLUE, color=BLUE, size=5.1)
    arrow(ax, (0.69, 0.245), (0.69, 0.220), color=NAVY)
    arrow(ax, (0.68, 0.155), (0.68, 0.110), color=NAVY)
    chip(ax, 0.54, 0.055, 0.31, 0.055, "QUALIFIED POLICY", face=WHITE, edge=NAVY, color=NAVY, size=5.6)


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )
    panels = (
        ("architecture-scenarios.pdf", racing_scenarios_panel),
        ("architecture-expert.pdf", expert_panel),
        ("architecture-imitation.pdf", imitation_panel),
        ("architecture-reinforcement.pdf", reinforcement_panel),
    )
    for filename, draw_panel in panels:
        figure, axis = plt.subplots(figsize=(3.0, 3.9))
        figure.patch.set_facecolor(WHITE)
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        draw_panel(axis)
        output_path = OUTPUT_DIR / filename
        figure.savefig(output_path, bbox_inches=None, pad_inches=0.0)
        plt.close(figure)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
