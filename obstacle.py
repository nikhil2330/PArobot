#!/usr/bin/env python3
"""
Obstacle-avoidance test (LiDAR only) — matplotlib-only (NO cv2 / imshow).

Behavior:
- Drives forward constantly.
- If LiDAR sees a close “cluster” in the front sector, it swerves away.
- When obstacle disappears, it "re-centers straight" by tank-turning back for a max time.

Center ignore wedge:
- Ignores (follow cone ±6°) + extra ±9° padding  => ignore ±15° (30° total) around 0°
  so you can simulate "person zone" not being treated as obstacle.

Stop: Ctrl+C
"""

import math
import time

import numpy as np
import matplotlib.pyplot as plt

from gpiozero import PWMLED, LED
import ydlidar


RANGE_MIN      = 0.15
RANGE_MAX      = 5.00
PLOT_MAX_RANGE = 4.0

ACCEL_LINEAR = 80.0
ACCEL_TURN   = 70.0
MIN_INNER_RATIO = 0.55

FORWARD_SPEED_TARGET = 30.0  # forward speed for this test

OBS_AVOID_DIST     = 0.90
OBS_BIAS_TURN      = 25.0
OBS_MIN_PTS        = 8

REACQUIRE_TIME_SEC = 1.2     # max time to "turn back straight"
REACQUIRE_TURN_MAX = 25.0    # cap turn during recenter tank turn

OBS_SECTOR_HALF_W = math.radians(60.0)  # scan front +/-60deg

# --- ignore wedge = follow cone(±6°) + extra 9° padding => ±15° ---
CONE_HALF_W = math.radians(6.0)
OBS_IGNORE_HALF_W = CONE_HALF_W + math.radians(9.0)  # 15°


# Motor inversion flags (keep yours)
INV_LEFT  = False
INV_RIGHT = False


# Left
l_rpwm = PWMLED("BOARD16", frequency=200)  # GPIO13 -> RPWM (Left)
l_lpwm = PWMLED("BOARD18", frequency=200)  # GPIO6  -> LPWM (Left)
l_en   = LED("BOARD22")                    # GPIO5  -> R_EN & L_EN

# Right
r_rpwm = PWMLED("BOARD35", frequency=200)  # GPIO19 -> LPWM (Right)
r_lpwm = PWMLED("BOARD37", frequency=200)  # GPIO26 -> RPWM (Right)
r_en   = LED("BOARD32")                    # GPIO12 -> R_EN & L_EN


def enable_all():
    l_en.on()
    r_en.on()


def disable_all():
    l_en.off()
    r_en.off()


def _drive_side(rpwm, lpwm, val):
    if val > 0:
        lpwm.value = 0
        rpwm.value = val
    elif val < 0:
        rpwm.value = 0
        lpwm.value = -val
    else:
        rpwm.value = 0
        lpwm.value = 0


def tank(left, right):
    """
    left, right in [-100..100]
    Positive = forward, negative = backward.
    """
    left  = max(-100.0, min(100.0, float(left)))  / 100.0
    right = max(-100.0, min(100.0, float(right))) / 100.0

    if INV_LEFT:
        left = -left
    if INV_RIGHT:
        right = -right

    _drive_side(l_rpwm, l_lpwm, left)
    _drive_side(r_rpwm, r_lpwm, right)


def smooth(prev, target, accel_per_sec, dt):
    max_step = accel_per_sec * dt
    delta = np.clip(target - prev, -max_step, +max_step)
    return prev + delta


def mix_tanky(forward, turn, min_inner_ratio=MIN_INNER_RATIO):
    left  = forward - turn
    right = forward + turn

    m = max(100.0, abs(left), abs(right))
    left  = left  * 100.0 / m
    right = right * 100.0 / m

    if abs(forward) > 5.0 and abs(turn) > 5.0:
        min_inner = min_inner_ratio * abs(forward)
        if abs(left) < abs(right):
            if abs(left) < min_inner:
                left = math.copysign(min_inner, left)
        else:
            if abs(right) < min_inner:
                right = math.copysign(min_inner, right)

    return float(np.clip(left, -100, 100)), float(np.clip(right, -100, 100))


# ---------------- LiDAR ----------------
laser = None
scan = ydlidar.LaserScan()


def init_lidar(port="/dev/ttyUSB0", baud=115200):
    global laser
    ydlidar.os_init()
    laser = ydlidar.CYdLidar()
    laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, baud)
    laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
    laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
    laser.setlidaropt(ydlidar.LidarPropScanFrequency, 6.0)
    laser.setlidaropt(ydlidar.LidarPropSampleRate, 4)
    laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)

    if not laser.initialize():
        raise RuntimeError("Failed to initialize LiDAR")
    if not laser.turnOn():
        raise RuntimeError("Failed to start LiDAR scanning")


def shutdown_lidar():
    try:
        if laser is not None:
            laser.turnOff()
            laser.disconnecting()
    except Exception:
        pass


def read_obstacle_sector():
    """
    Returns:
      xs, ys (plot points in front sector, excluding ignored center wedge)
      obs_left_ct, obs_right_ct (close points <= OBS_AVOID_DIST)
      obs_min_dist (min range in sector, excluding ignored wedge)
    Angle convention:
      x = r*sin(a), y = r*cos(a)
      a > 0 => RIGHT side
      a < 0 => LEFT side
    """
    if laser is None:
        return [], [], 0, 0, None

    if not laser.doProcessSimple(scan):
        return [], [], 0, 0, None

    xs, ys = [], []
    obs_left_ct = 0
    obs_right_ct = 0
    obs_min_dist = None

    for p in scan.points:
        r = p.range
        a = p.angle

        if not (RANGE_MIN < r <= RANGE_MAX):
            continue

        # front sector
        if (-OBS_SECTOR_HALF_W) <= a <= (+OBS_SECTOR_HALF_W):
            # ignore center wedge (simulate "person zone")
            if abs(a) <= OBS_IGNORE_HALF_W:
                continue

            if (obs_min_dist is None) or (r < obs_min_dist):
                obs_min_dist = r

            if r <= PLOT_MAX_RANGE:
                xs.append(r * math.sin(a))
                ys.append(r * math.cos(a))

            if r <= OBS_AVOID_DIST:
                if a >= 0:
                    obs_right_ct += 1
                else:
                    obs_left_ct += 1

    return xs, ys, obs_left_ct, obs_right_ct, obs_min_dist


# ---------------- Plot ----------------
fig = None
ax = None
pts_plot = None
sector_fill = None
txt_status = None


def init_plot():
    global fig, ax, pts_plot, sector_fill, txt_status
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(0.0, 3.0)
    ax.grid(True)

    pts_plot, = ax.plot([], [], ".", markersize=3, label="LiDAR sector points")
    ax.plot(0, 0, "ro", markersize=6, label="Robot")

    max_r = PLOT_MAX_RANGE
    left_x  = max_r * math.sin(-OBS_SECTOR_HALF_W)
    left_y  = max_r * math.cos(-OBS_SECTOR_HALF_W)
    right_x = max_r * math.sin(+OBS_SECTOR_HALF_W)
    right_y = max_r * math.cos(+OBS_SECTOR_HALF_W)

    sector_fill = ax.fill(
        [0, left_x, right_x],
        [0, left_y, right_y],
        "orange", alpha=0.2, label="Front sector")[0]

    # draw the ignored wedge boundaries (optional but helpful)
    ig = OBS_IGNORE_HALF_W
    ig_lx = max_r * math.sin(-ig)
    ig_ly = max_r * math.cos(-ig)
    ig_rx = max_r * math.sin(+ig)
    ig_ry = max_r * math.cos(+ig)
    ax.plot([0, ig_lx], [0, ig_ly], "--", linewidth=1.2, label="Ignore wedge")
    ax.plot([0, ig_rx], [0, ig_ry], "--", linewidth=1.2)

    txt_status = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left"
    )

    ax.legend(loc="upper right")


def update_plot(xs, ys, mode, obs_left, obs_right, obs_min, forward_cmd, turn_cmd, left_cmd, right_cmd):
    if fig is None:
        return

    pts_plot.set_data(xs, ys)

    if mode == "AVOID":
        sector_fill.set_facecolor("red")
    elif mode == "RECENTER":
        sector_fill.set_facecolor("orange")
    else:
        sector_fill.set_facecolor("green")

    txt_status.set_text(
        f"mode={mode}\n"
        f"obs_left={obs_left} obs_right={obs_right}\n"
        f"obs_min={obs_min if obs_min is not None else 0.0:.2f}\n"
        f"cmd F={forward_cmd:.1f} T={turn_cmd:.1f}\n"
        f"out L={left_cmd:.1f} R={right_cmd:.1f}\n"
        f"ignored_half={math.degrees(OBS_IGNORE_HALF_W):.1f} deg\n"
        f"Stop: Ctrl+C"
    )

    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    print("Initializing LiDAR...")
    init_lidar(port="/dev/ttyUSB0", baud=115200)

    print("Initializing plot...")
    init_plot()

    print("Enabling motors...")
    enable_all()
    time.sleep(0.1)

    forward_cmd = 0.0
    turn_cmd = 0.0
    last_time = time.time()

    # recenter state
    last_avoid_dir = 0   # +1 means we were turning LEFT, -1 means turning RIGHT
    recenter_start_time = None

    try:
        while True:
            now = time.time()
            dt = max(0.0, min(now - last_time, 0.2))
            last_time = now

            xs, ys, obs_left_ct, obs_right_ct, obs_min_dist = read_obstacle_sector()

            close_pts = obs_left_ct + obs_right_ct
            obstacle_now = (obs_min_dist is not None and obs_min_dist <= OBS_AVOID_DIST and close_pts >= OBS_MIN_PTS)

            target_forward_cmd = FORWARD_SPEED_TARGET
            target_turn_cmd = 0.0
            mode = "CLEAR"

            if obstacle_now:
                recenter_start_time = None

                if obs_right_ct > obs_left_ct:
                    target_turn_cmd = +OBS_BIAS_TURN   # obstacle right -> turn left
                    last_avoid_dir = +1
                else:
                    target_turn_cmd = -OBS_BIAS_TURN   # obstacle left -> turn right
                    last_avoid_dir = -1

                mode = "AVOID"

            else:
                if last_avoid_dir != 0:
                    if recenter_start_time is None:
                        recenter_start_time = now

                    if (now - recenter_start_time) <= REACQUIRE_TIME_SEC:
                        target_forward_cmd = 0.0  # tank turn back
                        target_turn_cmd = float(
                            np.clip(
                                -last_avoid_dir * REACQUIRE_TURN_MAX,
                                -REACQUIRE_TURN_MAX, +REACQUIRE_TURN_MAX
                            )
                        )
                        mode = "RECENTER"
                    else:
                        last_avoid_dir = 0
                        recenter_start_time = None
                        mode = "CLEAR"

            forward_cmd = smooth(forward_cmd, target_forward_cmd, ACCEL_LINEAR, dt)
            turn_cmd    = smooth(turn_cmd,    target_turn_cmd,    ACCEL_TURN,   dt)

            left_out, right_out = mix_tanky(forward_cmd, turn_cmd)
            tank(left_out, right_out)

            update_plot(
                xs, ys, mode,
                obs_left_ct, obs_right_ct, obs_min_dist,
                forward_cmd, turn_cmd,
                left_out, right_out
            )
            plt.pause(0.001)

    except KeyboardInterrupt:
        print("\nCtrl+C -> exiting...")

    finally:
        print("Shutting down...")
        try:
            tank(0, 0)
            disable_all()
        except Exception:
            pass
        try:
            shutdown_lidar()
        except Exception:
            pass


if __name__ == "__main__":
    main()
