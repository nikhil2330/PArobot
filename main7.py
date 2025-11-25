#!/usr/bin/env python3

import os
import math
import time
import argparse
import importlib.util
import serial  # <--- NEW

import numpy as np
import cv2
import matplotlib.pyplot as plt

from gpiozero import PWMLED, LED
from picamera2 import Picamera2
import ydlidar
from threading import Thread


CAMERA_FOV_DEG = 60.0  # camera fov
CAMERA_FOV_RAD = math.radians(CAMERA_FOV_DEG)

TURN_SPEED_TARGET    = 25.0   # target turn speed (percent)
FORWARD_SPEED_TARGET = 30.0   # target linear speed (percent)
CENTER_DEADZONE      = 0.10   # value where side to side jitter decreases

ACCEL_LINEAR = 200.0   # how fast forward/back can change (percent/sec)
ACCEL_TURN   = 200.0   # how fast turn can change (percent/sec)

FOLLOW_NEAR = 0.8      # too close -> back up
FOLLOW_FAR  = 1.5      # too far   -> go forward

RANGE_MIN      = 0.15
RANGE_MAX      = 5.00
CONE_HALF_W    = math.radians(6.0)   # LiDAR cone half-width around aim angle
PLOT_MAX_RANGE = 4.0                 # max range to show in LiDAR plot

# Motor inversion flags
INV_LEFT  = False
INV_RIGHT = False

# Detection / lock-on thresholds
DETECTION_THRESHOLD   = 0.5  # min confidence for person class
LOCK_VISIBLE_TIME_SEC = 2.0   # must be continuously visible this long
MIN_HIST_FRAMES       = 5     # minimum frames to build histogram
COLOR_SIM_THRESH      = 0.5   # histogram correlation threshold (0..1)
HIST_BINS             = 16    # H/S histogram bins

MAX_TRACK_LOST_FRAMES = 5  # max consecutive frames allowed to lose track


SERIAL_PORT = "/dev/ttyUSB1" 
BAUD_RATE   = 9600

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    print(f"[RF] Opened serial port {SERIAL_PORT}")
except Exception as e:
    print(f"[RF] Could not open serial port {SERIAL_PORT}: {e}")
    ser = None

robot_enabled = False  # global follow/stop flag


def reset_motion_state_hw():
    """
    Hardware-level reset: stop motors immediately.
    (State variables are reset inside main() on START edge.)
    """
    try:
        tank(0, 0)
    except Exception:
        pass


def rf_serial():
    global robot_enabled
    if ser is None:
        return

    try:
        while ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            print(f"[RF] {line}")
            if line == "START":
                robot_enabled = True
                reset_motion_state_hw()
                print("[RF] Robot started")
            elif line == "STOP":
                robot_enabled = False
                reset_motion_state_hw()
                print("[RF] Robot stopped")
    except Exception as e:
        print(f"[RF] Serial error: {e}")



# Left
l_rpwm = PWMLED("BOARD33", frequency=200)  # GPIO13 -> RPWM (Left)
l_lpwm = PWMLED("BOARD31", frequency=200)  # GPIO6  -> LPWM (Left)
l_en   = LED("BOARD29")                    # GPIO5  -> R_EN & L_EN

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
    Simple tank drive:
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



class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": resolution, "format": "RGB888"},
            lores={"size": resolution}
        )
        self.picam2.configure(config)
        self.picam2.start()

        self.frame = None
        self.stopped = False
        self.thread = Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            self.frame = self.picam2.capture_array("main")

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        try:
            self.picam2.close()
        except Exception:
            pass


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


def read_lidar_cone(aim_angle, cone_half_width=CONE_HALF_W, plot_max_range=PLOT_MAX_RANGE):
    if laser is None:
        return None, [], []

    if not laser.doProcessSimple(scan):
        return None, [], []

    distances = []
    xs, ys = [], []

    for p in scan.points:
        if RANGE_MIN < p.range <= RANGE_MAX:
            if (aim_angle - cone_half_width) <= p.angle <= (aim_angle + cone_half_width):
                distances.append(p.range)
                if p.range <= plot_max_range:
                    xs.append(p.range * math.sin(p.angle))
                    ys.append(p.range * math.cos(p.angle))

    front_distance = np.median(distances) if distances else None
    return front_distance, xs, ys



interpreter = None
input_details = None
output_details = None
labels = []
height = 0
width = 0
floating_model = False
boxes_idx = 0
classes_idx = 1
scores_idx = 2
input_mean = 127.5
input_std = 127.5


def init_tflite(model_dir, graph_name, labels_name, use_tpu=False):
    global interpreter, input_details, output_details
    global labels, height, width, floating_model
    global boxes_idx, classes_idx, scores_idx

    cwd = os.getcwd()
    path_to_ckpt  = os.path.join(cwd, model_dir, graph_name)
    path_to_label = os.path.join(cwd, model_dir, labels_name)

    # Labels
    with open(path_to_label, "r") as f:
        labels_local = [line.strip() for line in f.readlines()]
    if labels_local and labels_local[0] == "???":
        labels_local.pop(0)
    labels[:] = labels_local

    pkg = importlib.util.find_spec("tflite_runtime")
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_tpu:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_tpu:
            from tensorflow.lite.python.interpreter import load_delegate

    if use_tpu:
        if graph_name == "detect.tflite":
            graph_name = "edgetpu.tflite"
        path_to_ckpt = os.path.join(cwd, model_dir, graph_name)
        interpreter = Interpreter(
            model_path=path_to_ckpt,
            experimental_delegates=[load_delegate("libedgetpu.so.1.0")]
        )
    else:
        interpreter = Interpreter(model_path=path_to_ckpt)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    height = input_details[0]["shape"][1]
    width  = input_details[0]["shape"][2]
    floating_model = (input_details[0]["dtype"] == np.float32)

    outname = output_details[0]["name"]
    if "StatefulPartitionedCall" in outname:   # TF2
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:                                      # TF1
        boxes_idx, classes_idx, scores_idx = 0, 1, 2


def detect_people(frame_rgb, imW, imH, threshold):
    if interpreter is None:
        return []

    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    boxes   = interpreter.get_tensor(output_details[boxes_idx]["index"])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]["index"])[0]
    scores  = interpreter.get_tensor(output_details[scores_idx]["index"])[0]

    people = []
    for i, score in enumerate(scores):
        if threshold < score <= 1.0:
            cls_id = int(classes[i])
            label = labels[cls_id] if cls_id < len(labels) else "obj"
            if label == "person":
                ymin = int(max(0,   boxes[i][0] * imH))
                xmin = int(max(0,   boxes[i][1] * imW))
                ymax = int(min(imH, boxes[i][2] * imH))
                xmax = int(min(imW, boxes[i][3] * imW))
                people.append({
                    "bbox": (xmin, ymin, xmax, ymax),
                    "score": float(score),
                })

    return people


def compute_color_hist(frame_rgb, bbox):
    xmin, ymin, xmax, ymax = bbox
    h, w, _ = frame_rgb.shape

    # clamp
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w - 1, xmax)
    ymax = min(h - 1, ymax)

    if xmax <= xmin or ymax <= ymin:
        return None

    # crop to inner region to avoid background / edges
    bw = xmax - xmin
    bh = ymax - ymin
    if bw < 10 or bh < 10:
        return None  # too small, not reliable

    # shrink bbox: keep central area
    inner_xmin = int(xmin + 0.15 * bw)
    inner_xmax = int(xmax - 0.15 * bw)
    inner_ymin = int(ymin + 0.20 * bh)  # skip more of head/feet
    inner_ymax = int(ymax - 0.05 * bh)

    inner_xmin = max(xmin, inner_xmin)
    inner_ymin = max(ymin, inner_ymin)
    inner_xmax = min(xmax, inner_xmax)
    inner_ymax = min(ymax, inner_ymax)

    if inner_xmax <= inner_xmin or inner_ymax <= inner_ymin:
        return None

    roi = frame_rgb[inner_ymin:inner_ymax, inner_xmin:inner_xmax]
    if roi.size == 0:
        return None

    # small blur to reduce noise
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)

    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_RGB2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1],          # H and S channels
        None,
        [HIST_BINS, HIST_BINS],
        [0, 180, 0, 256]
    )
    cv2.normalize(hist, hist)
    return hist.astype(np.float32)


def hist_correlation(hist1, hist2):
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))


fig = None
ax = None
lidar_points_plot = None
fov_fill = None
fov_left_line = None
fov_right_line = None


def init_plot():
    global fig, ax, lidar_points_plot, fov_fill, fov_left_line, fov_right_line
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(0.0, 3.0)
    ax.grid(True)

    lidar_points_plot, = ax.plot([], [], ".", markersize=3, label="LiDAR points")
    ax.plot(0, 0, "ro", markersize=6, label="Robot")

    fov_fill = ax.fill([], [], "orange", alpha=0.2, label="LiDAR cone")[0]
    fov_left_line,  = ax.plot([], [], "orange", linewidth=1.5)
    fov_right_line, = ax.plot([], [], "orange", linewidth=1.5)

    ax.legend(loc="upper right")


def update_plot(aim_angle, xs, ys, detected):
    if fig is None:
        return

    lidar_points_plot.set_data(xs, ys)

    cone = CONE_HALF_W
    max_r = PLOT_MAX_RANGE

    left_x = [0, max_r * math.sin(aim_angle - cone)]
    left_y = [0, max_r * math.cos(aim_angle - cone)]
    right_x = [0, max_r * math.sin(aim_angle + cone)]
    right_y = [0, max_r * math.cos(aim_angle + cone)]

    cone_fill_x = [
        0,
        max_r * math.sin(aim_angle - cone),
        max_r * math.sin(aim_angle + cone),
    ]
    cone_fill_y = [
        0,
        max_r * math.cos(aim_angle - cone),
        max_r * math.cos(aim_angle + cone),
    ]
    fov_fill.set_xy(np.column_stack((cone_fill_x, cone_fill_y)))

    color = "green" if detected else "red"
    fov_fill.set_facecolor(color)
    fov_left_line.set_data(left_x, left_y)
    fov_right_line.set_data(right_x, right_y)
    fov_left_line.set_color(color)
    fov_right_line.set_color(color)

    fig.canvas.draw()
    fig.canvas.flush_events()




def smooth(prev, target, accel_per_sec, dt):
    max_step = accel_per_sec * dt
    delta = np.clip(target - prev, -max_step, +max_step)
    return prev + delta



def main():
    global robot_enabled

    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldir", required=True,
                        help="Folder where .tflite and labelmap live")
    parser.add_argument("--graph", default="detect.tflite",
                        help="TFLite model filename")
    parser.add_argument("--labels", default="labelmap.txt",
                        help="Label map filename")
    parser.add_argument("--threshold", type=float, default=DETECTION_THRESHOLD,
                        help="Min confidence for person detection")
    parser.add_argument("--resolution", default="640x480",
                        help="Camera resolution WxH (e.g., 640x480)")
    parser.add_argument("--edgetpu", action="store_true",
                        help="Use Coral Edge TPU model")
    args = parser.parse_args()

    resW, resH = args.resolution.split("x")
    imW, imH = int(resW), int(resH)

    print("Initializing TFLite model...")
    init_tflite(args.modeldir, args.graph, args.labels, use_tpu=args.edgetpu)

    print("Initializing LiDAR...")
    init_lidar(port="/dev/ttyUSB0", baud=115200)

    print("Initializing plots...")
    init_plot()

    print("Starting Picamera2 stream...")
    videostream = VideoStream(resolution=(imW, imH), framerate=30)

    enable_all()
    time.sleep(0.1)

    freq = cv2.getTickFrequency()

    candidate_active          = False
    candidate_bbox            = None
    candidate_hist_accum      = None
    candidate_hist_count      = 0
    candidate_visible_time    = 0.0
    candidate_visible_start_t = None

    tracked_active      = False
    tracked_bbox        = None
    tracked_hist        = None
    tracked_score       = 0.0
    tracked_lost_frames = 0

    forward_cmd = 0.0   # actual command sent to motors
    turn_cmd    = 0.0
    last_time   = time.time()

    prev_robot_enabled = robot_enabled

    print(f"Stay in view for ~{LOCK_VISIBLE_TIME_SEC} seconds to lock person")

    try:
        while True:
            t1 = cv2.getTickCount()
            now = time.time()
            dt = now - last_time
            last_time = now
            dt = max(0.0, min(dt, 0.2))  # clamp dt

            # --- Read RF commands ---
            rf_serial()

            frame = videostream.read()
            if frame is None:
                continue

            if robot_enabled and not prev_robot_enabled:
                candidate_active          = False
                candidate_bbox            = None
                candidate_hist_accum      = None
                candidate_hist_count      = 0
                candidate_visible_time    = 0.0
                candidate_visible_start_t = None

                tracked_active      = False
                tracked_bbox        = None
                tracked_hist        = None
                tracked_score       = 0.0
                tracked_lost_frames = 0

                forward_cmd = 0.0
                turn_cmd    = 0.0
                tank(0, 0)
                print("[RF] New Start")
            prev_robot_enabled = robot_enabled

            frame_rgb = frame

            people = detect_people(frame_rgb, imW, imH, args.threshold)

            state_str = "NO_PERSON"
            candidate_visible_this_frame = False


            if not tracked_active:
                for det in people:
                    x1, y1, x2, y2 = det["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

                if not candidate_active:
                    if people:
                        best = max(people, key=lambda d: d["score"])
                        candidate_bbox = best["bbox"]
                        candidate_hist_accum = None
                        candidate_hist_count = 0
                        candidate_visible_time = 0.0
                        candidate_visible_start_t = None
                        candidate_active = True

                        print("Person selected for locking")
                        state_str = "CANDIDATE_STARTED"
                    else:
                        state_str = "WAITING_FOR_PERSON"
                else:
                    # Candidate already chosen
                    if people:
                        best = max(people, key=lambda d: d["score"])
                        candidate_bbox = best["bbox"]

                        hist = compute_color_hist(frame_rgb, candidate_bbox)
                        if hist is not None:
                            if candidate_hist_accum is None:
                                candidate_hist_accum = hist.copy()
                                candidate_hist_count = 1
                            else:
                                candidate_hist_accum += hist
                                candidate_hist_count += 1

                        candidate_visible_this_frame = True
                        state_str = "CANDIDATE_VISIBLE"
                    else:
                        candidate_visible_this_frame = False
                        candidate_visible_start_t = None
                        candidate_visible_time = 0.0
                        
                        state_str = "CANDIDATE_NOT_VISIBLE"

                    # Continuous visibility timing
                    if candidate_visible_this_frame:
                        if candidate_visible_start_t is None:
                            candidate_visible_start_t = now
                            candidate_visible_time = 0.0
                        else:
                            candidate_visible_time = now - candidate_visible_start_t

                    # Draw candidate bbox (blue)
                    if candidate_visible_this_frame and candidate_bbox is not None:
                        x1, y1, x2, y2 = candidate_bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Check lock condition
                    if (candidate_visible_this_frame and
                        candidate_hist_accum is not None and
                        candidate_hist_count >= MIN_HIST_FRAMES and
                        candidate_visible_time >= LOCK_VISIBLE_TIME_SEC):
                        avg_hist = candidate_hist_accum / max(1, candidate_hist_count)
                        cv2.normalize(avg_hist, avg_hist)
                        tracked_hist = avg_hist.astype(np.float32)
                        tracked_bbox = candidate_bbox
                        tracked_score = 0.0
                        tracked_active = True
                        tracked_lost_frames = 0

                        print("LOCKED onto person")
                        state_str = "LOCKED"
            else:
                best_match = None
                best_sim = None

                if people and tracked_hist is not None:
                    for det in people:
                        bbox = det["bbox"]
                        hist = compute_color_hist(frame_rgb, bbox)
                        if hist is None:
                            continue

                        color_sim = hist_correlation(tracked_hist, hist)
                        if color_sim < COLOR_SIM_THRESH:
                            continue

                        if (best_sim is None) or (color_sim > best_sim):
                            best_sim = color_sim
                            best_match = det

                if best_match is not None:
                    tracked_bbox = best_match["bbox"]
                    tracked_score = best_match["score"]
                    tracked_lost_frames = 0
                    state_str = f"LOCKED (sim={best_sim:.2f})"
                else:
                    tracked_lost_frames += 1
                    if tracked_lost_frames > MAX_TRACK_LOST_FRAMES:
                        # Stop using stale bbox so robot stops moving.
                        # DO NOT reset tracked_active → no re-locking.
                        tracked_bbox = None
                        state_str = f"LOCKED_LOST_STOP ({tracked_lost_frames})"
                    else:
                        state_str = f"LOCKED_LOST ({tracked_lost_frames})"

                if tracked_bbox is not None:
                    x1, y1, x2, y2 = tracked_bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"LOCKED score={tracked_score:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

            target_forward_cmd = 0.0
            target_turn_cmd = 0.0
            offset_norm = 0.0
            aim_angle = 0.0

            has_lock_and_bbox = tracked_active and (tracked_bbox is not None)

            if has_lock_and_bbox:
                xmin, ymin, xmax, ymax = tracked_bbox

                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0

                label = f"person {tracked_score:.2f} ({int(x_center)},{int(y_center)})"
                cv2.putText(
                    frame,
                    label,
                    (xmin, max(0, ymin - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                frame_center_x = imW / 2.0
                offset_norm = (x_center - frame_center_x) / frame_center_x

                aim_angle = offset_norm * (CAMERA_FOV_RAD / 2.0)

                if offset_norm < -CENTER_DEADZONE:
                    target_turn_cmd = +TURN_SPEED_TARGET
                elif offset_norm > +CENTER_DEADZONE:
                    target_turn_cmd = -TURN_SPEED_TARGET
                else:
                    target_turn_cmd = 0.0

            front_distance, xs, ys = read_lidar_cone(aim_angle)

            if has_lock_and_bbox and (front_distance is not None):
                cv2.putText(
                    frame,
                    f"Dist: {front_distance:.2f} m",
                    (30, imH - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                if front_distance > FOLLOW_FAR:
                    target_forward_cmd = +FORWARD_SPEED_TARGET
                    state_str = state_str + " | FORWARD"
                elif front_distance < FOLLOW_NEAR:
                    target_forward_cmd = -FORWARD_SPEED_TARGET
                    state_str = state_str + " | BACKWARD"
                else:
                    target_forward_cmd = 0.0
                    state_str = state_str + " | HOLD_DIST"
            else:
                target_forward_cmd = 0.0

            if not has_lock_and_bbox:
                target_forward_cmd = 0.0
                target_turn_cmd = 0.0

            if robot_enabled:
                forward_cmd = smooth(forward_cmd, target_forward_cmd, ACCEL_LINEAR, dt)
                turn_cmd    = smooth(turn_cmd,    target_turn_cmd,   ACCEL_TURN,   dt)

                left  = np.clip(forward_cmd - turn_cmd, -100, 100)
                right = np.clip(forward_cmd + turn_cmd, -100, 100)
                tank(left, right)

                motion_str = ""
                if forward_cmd > 1.0:
                    if turn_cmd > 1.0:
                        motion_str = "FORWARD + TURN LEFT"
                    elif turn_cmd < -1.0:
                        motion_str = "FORWARD + TURN RIGHT"
                    else:
                        motion_str = "FORWARD STRAIGHT"
                elif forward_cmd < -1.0:
                    if turn_cmd > 1.0:
                        motion_str = "BACKWARD + TURN LEFT"
                    elif turn_cmd < -1.0:
                        motion_str = "BACKWARD + TURN RIGHT"
                    else:
                        motion_str = "BACKWARD STRAIGHT"
                else:
                    if turn_cmd > 1.0:
                        motion_str = "TURN LEFT IN PLACE"
                    elif turn_cmd < -1.0:
                        motion_str = "TURN RIGHT IN PLACE"
                    else:
                        motion_str = "STOP"

                print(
                    f"[MOTION] {motion_str} | state={state_str} | "
                    f"dist={front_distance if front_distance is not None else 'None'} | "
                    f"L={left:.1f} R={right:.1f}"
                )
            else:
                # RF STOP: hard stop motors, keep commands at 0
                forward_cmd = 0.0
                turn_cmd    = 0.0
                tank(0, 0)
                state_str = state_str + " | RF_DISABLED"

            # ---- Overlays ----
            t2 = cv2.getTickCount()
            frame_rate_calc = 1.0 / ((t2 - t1) / freq)

            cv2.putText(
                frame,
                f"FPS: {frame_rate_calc:.2f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"STATE: {state_str}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"offset: {offset_norm:.2f}",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )
            cv2.putText(
                frame,
                f"RF: {'ON' if robot_enabled else 'OFF'}",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            if not tracked_active and candidate_active:
                cv2.putText(
                    frame,
                    f"VISIBLE: {candidate_visible_time:.2f}s / {LOCK_VISIBLE_TIME_SEC:.2f}s",
                    (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            update_plot(aim_angle, xs, ys, has_lock_and_bbox)

            cv2.imshow("Person Follow + Lock", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

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
        try:
            videostream.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
