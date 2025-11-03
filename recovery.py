#!/usr/bin/env python3
# Minimal recovery-behavior exerciser (no camera / no TFLite)
# Uses the same BTS7960 pins and LiDAR windows as main.py.

import time, math, numpy as np
from gpiozero import LED, PWMLED
import ydlidar

# ----------------- BTS7960 PIN MAP (BOARD numbering) -----------------
l_rpwm = PWMLED("BOARD33", frequency=200)  # GPIO13 -> RPWM (Left)
l_lpwm = PWMLED("BOARD31", frequency=200)  # GPIO6  -> LPWM (Left)
l_en   = LED("BOARD29")                    # GPIO5  -> R_EN & L_EN

r_rpwm = PWMLED("BOARD35", frequency=200)  # GPIO19 -> RPWM (Right)
r_lpwm = PWMLED("BOARD37", frequency=200)  # GPIO26 -> LPWM (Right)
r_en   = LED("BOARD32")                    # GPIO12 -> R_EN & L_EN

INV_LEFT  = False
INV_RIGHT = False

def enable_all(): l_en.on(); r_en.on()
def disable_all(): l_en.off(); r_en.off()

MIN_PWM = 0.32
def _apply_deadband(val):
    if val == 0.0: return 0.0
    s = 1 if val > 0 else -1
    a = abs(val)
    return s * (MIN_PWM + (1.0 - MIN_PWM) * a)

def _drive_side(rpwm, lpwm, val):
    if val > 0:
        lpwm.value = 0; rpwm.value = val
    elif val < 0:
        rpwm.value = 0; lpwm.value = -val
    else:
        rpwm.value = 0; lpwm.value = 0

def tank(l, r):
    l = max(-100.0, min(100.0, float(l))) / 100.0
    r = max(-100.0, min(100.0, float(r))) / 100.0
    if INV_LEFT:  l = -l
    if INV_RIGHT: r = -r
    l = _apply_deadband(l); r = _apply_deadband(r)
    _drive_side(l_rpwm, l_lpwm, l); _drive_side(r_rpwm, r_lpwm, r)

# LiDAR init
ydlidar.os_init()
laser = ydlidar.CYdLidar()
laser.setlidaropt(ydlidar.LidarPropSerialPort, "/dev/ttyUSB0")
laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 115200)
laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
laser.setlidaropt(ydlidar.LidarPropScanFrequency, 6.0)
laser.setlidaropt(ydlidar.LidarPropSampleRate, 4)
laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)

if not laser.initialize():
    print("Failed to initialize LiDAR"); exit(1)
if not laser.turnOn():
    print("Failed to start scanning"); exit(1)

scan = ydlidar.LaserScan()

# --- Parameters (mirror main.py) ---
CENTER_FOV   = math.radians(60.0)
FRONT_MARGIN = math.radians(6.0)
SIDE_OFFSET  = math.radians(20.0)
SIDE_SPREAD  = math.radians(40.0)

RANGE_MIN   = 0.15
RANGE_MAX   = 3.0

RECOVERY_TURN_SPEED = 48
RECOVERY_CREEP      = 32
SAFETY_MARGIN       = 0.3

CLEAR_DIST    = 1.0
CORNER_NEAR   = 0.70
STOP_CRITICAL = 0.35
SCAN_TIMEOUT  = 4.0

# --- State ---
front_distance = None
left_min = None
right_min = None
last_seen_angle = 0.0
last_seen_dist  = 1.0

recovery_phase = -1  # start with scan; will switch to corner if near walls
scan_dir = 1
recovery_start = time.time()

def read_lidar(aim_angle):
    global front_distance, left_min, right_min
    if not laser.doProcessSimple(scan): return False
    a_min = aim_angle - FRONT_MARGIN
    a_max = aim_angle + FRONT_MARGIN
    target_points = [p.range for p in scan.points
                     if (a_min <= p.angle <= a_max) and (RANGE_MIN < p.range <= RANGE_MAX)]
    front_distance = np.median(sorted(target_points)[:3]) if target_points else None
    left_points = [p.range for p in scan.points
                   if (aim_angle + SIDE_OFFSET) < p.angle < (aim_angle + SIDE_OFFSET + SIDE_SPREAD)
                   and RANGE_MIN < p.range < RANGE_MAX]
    right_points = [p.range for p in scan.points
                    if (aim_angle - SIDE_OFFSET - SIDE_SPREAD) < p.angle < (aim_angle - SIDE_OFFSET)
                    and RANGE_MIN < p.range < RANGE_MAX]
    left_min  = min(left_points)  if left_points  else None
    right_min = min(right_points) if right_points else None
    return True

def environment_is_clear():
    f_ok = (front_distance is None) or (front_distance > CLEAR_DIST)
    l_ok = (left_min is None) or (left_min > CLEAR_DIST)
    r_ok = (right_min is None) or (right_min > CLEAR_DIST)
    return f_ok and l_ok and r_ok

def environment_is_corner():
    conds = []
    if front_distance is not None: conds.append(front_distance < CORNER_NEAR)
    if left_min is not None:       conds.append(left_min < CORNER_NEAR)
    if right_min is not None:      conds.append(right_min < CORNER_NEAR)
    return any(conds)

enable_all()
time.sleep(0.1)

try:
    last_update = time.time()
    while True:
        now = time.time()
        read_lidar(last_seen_angle)

        # dynamic switch between scan and corner modes
        if recovery_phase == -1 and environment_is_corner():
            recovery_phase = 0
            recovery_start = now
            print("SCAN → CORNER (phase=0)")
        if recovery_phase != -1 and environment_is_clear() and (front_distance is None or front_distance > CLEAR_DIST+0.2):
            # if we moved out into open, go back to scanning
            recovery_phase = -1
            scan_dir = 1 if last_seen_angle >= 0 else -1
            recovery_start = now
            print("CORNER → SCAN (phase=-1)")

        f = front_distance if front_distance is not None else 99.0
        l = left_min       if left_min       is not None else 99.0
        r = right_min      if right_min      is not None else 99.0

        target_forward = 0.0
        target_turn    = 0.0

        if recovery_phase == -1:
            # Open-area spin scan
            target_turn = scan_dir * RECOVERY_TURN_SPEED
            if (now - recovery_start) > SCAN_TIMEOUT:
                scan_dir *= -1
                recovery_start = now

        elif recovery_phase == 0:
            # Corner: turn away until front clears
            if f < 0.7:
                target_turn = RECOVERY_TURN_SPEED if l >= r else -RECOVERY_TURN_SPEED
            else:
                recovery_phase = 1
                recovery_start = now
                print("→ Phase 1: forward creep")

        elif recovery_phase == 1:
            # Corner: creep forward toward last seen distance
            target_forward = 0.0 if f < STOP_CRITICAL else RECOVERY_CREEP
            if (front_distance is not None) and (front_distance <= (last_seen_dist - SAFETY_MARGIN)):
                recovery_phase = 2
                recovery_start = now
                print("→ Phase 2: peek")

        elif recovery_phase == 2:
            # Corner: peek toward last-seen angle
            target_turn = RECOVERY_TURN_SPEED if last_seen_angle >= 0 else -RECOVERY_TURN_SPEED
            target_forward = RECOVERY_CREEP if f > 1.0 else 0.0

        left  = np.clip(target_forward - target_turn, -100, 100)
        right = np.clip(target_forward + target_turn, -100, 100)
        tank(left, right)

        print(f"phase={recovery_phase} | L:{left:.1f} R:{right:.1f} | fd={front_distance} lm={left_min} rm={right_min}")
        time.sleep(0.02)

except KeyboardInterrupt:
    pass
finally:
    try:
        tank(0,0); disable_all()
        laser.turnOff(); laser.disconnecting()
    except:
        pass
