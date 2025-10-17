import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrow
from matplotlib.animation import FuncAnimation
import math

# --- Parameters ---
TURN_GAIN = 20       # turning aggressiveness
FORWARD_GAIN = 0.4   # forward speed responsiveness
TARGET_DIST = 0.8    # desired follow distance (m)
CORRECTION = 0.3     # smoothing factor (0–1)
WHEEL_BASE = 0.25    # meters
DT = 0.1             # timestep (s)

# --- State ---
robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0
prev_f, prev_t = 0.0, 0.0

# --- Person motion (smooth side-to-side + in/out) ---
T = 600
t_vals = np.linspace(0, 60, T)
person_x = 1.2 * np.sin(t_vals / 3)
person_y = 1.0 + 0.4 * np.cos(t_vals / 8)

# --- Visualization ---
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-2, 2)
ax.set_ylim(-0.5, 3)
ax.set_aspect("equal")
ax.grid(True)

robot_dot, = ax.plot([], [], "ro", markersize=6)
person_dot, = ax.plot([], [], "go", markersize=6)
path_line, = ax.plot([], [], "r--", linewidth=1)
fov_wedge = Wedge((0, 0), 2.0, 0, 0, color="blue", alpha=0.25)
ax.add_patch(fov_wedge)
heading_arrow = FancyArrow(0, 0, 0, 0, width=0.05, color="orange")
ax.add_patch(heading_arrow)

trace_x, trace_y = [], []

# --- Update function ---
def update(frame):
    global robot_x, robot_y, robot_theta, prev_f, prev_t, heading_arrow

    # Person position
    px, py = person_x[frame], person_y[frame]

    # Relative vector to person (robot faces +Y)
    dx, dy = px - robot_x, py - robot_y
    distance = math.hypot(dx, dy)
    angle_to_person = math.atan2(dx, dy)  # angle relative to +Y axis

    # Angle error (difference between where robot faces and person)
    angle_error = (angle_to_person - robot_theta + math.pi) % (2 * math.pi) - math.pi
    turn_s = np.clip(TURN_GAIN * angle_error, -50, 50)

    # Forward speed based on distance
    diff = distance - TARGET_DIST
    forward_s = np.clip(FORWARD_GAIN * diff * 100.0, -60, 60)

    # Smooth speeds
    forward_s = prev_f * (1 - CORRECTION) + forward_s * CORRECTION
    turn_s = prev_t * (1 - CORRECTION) + turn_s * CORRECTION
    prev_f, prev_t = forward_s, turn_s

    # Differential-drive motion
    left = np.clip(forward_s - turn_s, -100, 100)
    right = np.clip(forward_s + turn_s, -100, 100)
    v = (left + right) / 200.0
    omega = (right - left) / (2 * WHEEL_BASE)

    # Update pose (robot faces +Y)
    robot_theta += omega * DT
    robot_x += v * math.sin(robot_theta) * DT
    robot_y += v * math.cos(robot_theta) * DT

    # --- Visualization updates ---
    robot_dot.set_data(robot_x, robot_y)
    person_dot.set_data(px, py)
    trace_x.append(robot_x)
    trace_y.append(robot_y)
    path_line.set_data(trace_x, trace_y)

    # FOV wedge
    fov_wedge.set_center((robot_x, robot_y))
    center_deg = math.degrees(robot_theta)
    fov_half = 10
    fov_wedge.set_theta1(center_deg - fov_half)
    fov_wedge.set_theta2(center_deg + fov_half)

    # Heading arrow
    heading_arrow.remove()
    arrow_len = 0.3
    heading_arrow = FancyArrow(robot_x, robot_y,
                               arrow_len * math.sin(robot_theta),
                               arrow_len * math.cos(robot_theta),
                               width=0.05, color="orange")
    ax.add_patch(heading_arrow)

    ax.set_title(f"Dist={distance:.2f}  Fwd={forward_s:.1f}  Turn={turn_s:.1f}")
    return robot_dot, person_dot, path_line, fov_wedge, heading_arrow

ani = FuncAnimation(fig, update, frames=T, interval=50, blit=True)
plt.show()
