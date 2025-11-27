######## Picam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os, sys, types
import serial
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from pydub import AudioSegment
from pydub.playback import play
from gpiozero import LED,PWMLED
from time import sleep
import ydlidar
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import threading
import time
from picamera2 import Picamera2


SERIAL_PORT = "/dev/ttyUSB1"   # change to /dev/ttyUSB0 or whatever your Arduino shows
BAUD_RATE = 9600

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    print(f"[RF] Opened serial port {SERIAL_PORT}")
except Exception as e:
    print(f"[RF] Could not open serial port {SERIAL_PORT}: {e}")
    ser = None  # run without RF if serial fails

robot_enabled = False  # start in STOP mode

# #song = AudioSegment.from_mp3("person.mp3")

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
                reset_motion_state() 
                print("[RF] Robot ENABLED (follow mode)")
            elif line == "STOP":
                robot_enabled = False
                reset_motion_state() 
                print("[RF] Robot DISABLED (motors stopped)")
    except Exception as e:
        print(f"[RF] Serial error: {e}")

# Left BTS7960
l_rpwm = PWMLED("BOARD33", frequency=200)  # GPIO13 -> RPWM (Left)
l_lpwm = PWMLED("BOARD31", frequency=200)  # GPIO6  -> LPWM (Left)
l_en   = LED("BOARD29")                    # GPIO5  -> R_EN & L_EN (tie both to this)

# Right BTS7960
r_rpwm = PWMLED("BOARD35", frequency=200)  # GPIO19 -> LPWM (Right)
r_lpwm = PWMLED("BOARD37", frequency=200)  # GPIO26 -> RPWM (Right)
r_en   = LED("BOARD32")                    # GPIO12 -> R_EN & L_EN (tie both to this)

INV_LEFT  = False
INV_RIGHT = False

# Safe default state
def enable_all():
    l_en.on()
    r_en.on()

MIN_PWM = 0.32  # tune per motor: 0.28–0.40 typical

def _apply_deadband(val):
    if val == 0.0:
        return 0.0
    s = 1 if val > 0 else -1
    a = abs(val)
    return s * (MIN_PWM + (1.0 - MIN_PWM) * a)

def disable_all():
    l_en.off()
    r_en.off()

def _drive_side(rpwm, lpwm, val):
    """val in [-1..1]; >0 uses RPWM, <0 uses LPWM, 0 coasts."""
    if val > 0:
        lpwm.value = 0
        rpwm.value = val
    elif val < 0:
        rpwm.value = 0
        lpwm.value = -val
    else:
        rpwm.value = 0
        lpwm.value = 0

# New tank() for BTS7960: accepts left/right in [-100..100]
def tank(l, r):
    # map from [-100..100] to [-1..1]
    l = max(-100.0, min(100.0, float(l))) / 100.0
    r = max(-100.0, min(100.0, float(r))) / 100.0
    if INV_LEFT:  l = -l
    if INV_RIGHT: r = -r
    l = _apply_deadband(l)
    r = _apply_deadband(r)
    _drive_side(l_rpwm, l_lpwm, l)
    _drive_side(r_rpwm, r_lpwm, r)

# --------------------------------------------------------------------------

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
    print("Failed to initialize LiDAR")
    exit(1)

if not laser.turnOn():
    print("Failed to start scanning")
    exit(1)

scan = ydlidar.LaserScan()

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": resolution, "format": "RGB888"},  
            lores={"size": resolution}
        )

        self.picam2.configure(config)
        self.picam2.start()
        self.frame = None
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
    # Start the thread that reads frames from the video stream
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            # Capture directly as RGB array
            frame = self.picam2.capture_array("main")
            self.frame = frame  # No cvtColor needed for RGB888

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True
        self.picam2.close()


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True, default='./')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu
print(imW,imH)
# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# imW, imH = 640, 480 
# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)
ctr=0
frame_count = 0

# --- Initialize plot ---
plt.ion()  # interactive mode
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1, 1)   # X-axis range (-1m to 1m)
ax.set_ylim(0, 3)    # Y-axis range (0m to 1m)
ax.grid(True)

lidar_points_plot, = ax.plot([], [], 'b.', markersize=3, label="LIDAR points")
robot_dot, = ax.plot(0, 0, 'ro', markersize=6, label="Robot")

# Add FOV cone (two boundary lines + fill)
fov_fill = ax.fill([], [], 'orange', alpha=0.2, label="Camera FOV")[0]
fov_left_line, = ax.plot([], [], 'orange', linewidth=1.5)
fov_right_line, = ax.plot([], [], 'orange', linewidth=1.5)
ax.legend(loc="upper right")

# --- CONTROL & BEHAVIOR CONSTANTS ---------------------------------------------

# --- Motion & turning ---
turn = 30                     # max turning PWM output (0..100)  slightly gentler cap
forward_speed = 38            # max forward PWM output (0..100)

# --- Behavior tuning ---
accel_rate_fwd  = 36.0        # forward ramp (PWM units/sec)

# Asymmetric turn ramps: slow to spool up, FAST to brake/stop
accel_rate_turn_up    = 22.0  # ramp when |target| is growing (gentle)      
accel_rate_turn_down  = 150.0 # ramp when braking / sign change (quick)     

alpha = 0.30                  # smoothing factor for person x-position (0-1)
dead_zone_ratio = 0.18        # keep modest; do NOT increase dead-zone itself
turn_sensitivity = 0.80       # proportional gain (slightly gentler)        
turn_D = 0.22                 # derivative gain (more damping)               
turn_hysteresis = 0.05        # tiny hysteresis band to avoid instant sign flip
zero_cross_band = 0.08        # extra brake band around center (helps weave) 

# Bias knobs to fix “pulls right/left”
CENTER_BIAS_PX = 0.0          # +ve shifts neutral point to the RIGHT (pixels) 
TURN_BIAS      = 0.0          # added to target_turn in PWM units (e.g., -2)    
LEFT_GAIN      = 1.00         # scale left side output (1.00 = no change)       
RIGHT_GAIN     = 1.00         # scale right side output                         

lost_timeout = 1.0            # sec - how long to remember last seen person (bbox reuse)

# --- Distance thresholds (in meters) ---
FOLLOW_FAR   = 1.20
FOLLOW_NEAR  = 0.80
SIDE_WARN    = 0.50
SIDE_STOP    = 0.30
RANGE_MIN    = 0.15
RANGE_MAX    = 4.00
CLOSE_RATE   = 0.30

# --- LIDAR cone geometry (radians) ---
CENTER_FOV   = math.radians(60.0)
FRONT_MARGIN = math.radians(6.0)
SIDE_OFFSET  = math.radians(20.0)
SIDE_SPREAD  = math.radians(40.0)

# --- Initialization for motion state ---
last_update = time.time()
last_ctrl_time = last_update   # separate timestamp for controller dt         
current_forward = 0.0
current_turn = 0.0

prev_dist = None
front_distance = None
left_min = None
right_min = None

smooth_x = None

last_person_box = None
last_confidence_time = 0

person_visible = False
last_seen_time = 0.0
last_seen_angle = 0.0
last_seen_dist = 1.0

SEARCH_SPIN_TURN = 8         # SLOWER spin as requested                      
SEARCH_FLIP_SEC  = 6.0        # flip spin direction every N seconds
STOP_PAUSE_SEC   = 1.5        # stand still before spinning
search_mode = False
search_stage = 0              # 0 = pause, 1 = spin
search_start = 0.0
scan_dir = 1                  # +1 = CCW, -1 = CW

prev_error = 0.0              # for derivative term


def reset_motion_state():
    global current_forward, current_turn, prev_error
    global search_mode, search_stage, smooth_x, prev_dist
    global last_update, last_ctrl_time

    # Zero all motion-related state
    current_forward = 0.0
    current_turn = 0.0
    prev_error = 0.0
    prev_dist = None
    smooth_x = None

    # Clear search behavior
    search_mode = False
    search_stage = 0

    # Reset timers so ramps start fresh
    now = time.time()
    last_update = now
    last_ctrl_time = now

    # Make sure motors are actually stopped
    tank(0, 0)

def read_lidar(aim_angle):
    """
    Refresh LiDAR-derived distances using a window centered at aim_angle.
    Updates globals: front_distance, left_min, right_min.
    Returns True if a scan was processed; False otherwise.
    """
    global front_distance, left_min, right_min
    if not laser.doProcessSimple(scan):
        return False

    # FRONT window (person-centered or last-seen angle)
    a_min = aim_angle - FRONT_MARGIN
    a_max = aim_angle + FRONT_MARGIN
    target_points = [p.range for p in scan.points
                     if (a_min <= p.angle <= a_max) and (RANGE_MIN < p.range <= RANGE_MAX)]
    front_distance = np.median(sorted(target_points)[:3]) if target_points else None

    # LEFT / RIGHT windows (for obstacle shaping)
    left_points = [p.range for p in scan.points
                   if (aim_angle + SIDE_OFFSET) < p.angle < (aim_angle + SIDE_OFFSET + SIDE_SPREAD)
                   and RANGE_MIN < p.range < RANGE_MAX]
    right_points = [p.range for p in scan.points
                    if (aim_angle - SIDE_OFFSET - SIDE_SPREAD) < p.angle < (aim_angle - SIDE_OFFSET)
                    and RANGE_MIN < p.range < RANGE_MAX]
    left_min  = min(left_points)  if left_points  else None
    right_min = min(right_points) if right_points else None
    return True

# --- MAIN CONTROL LOOP --------------------------------------------------------

# Enable motor drivers before loop
enable_all()
# Small delay to ensure BTS logic is live
time.sleep(0.1)

try:
    while True:
        frame_count += 1
        t1 = cv2.getTickCount()
        rf_serial()

        frame1 = videostream.read()
        if frame1 is None:
            continue
        frame = frame1.copy()

        # --- TensorFlow person detection ---
        frame_resized = cv2.resize(frame, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes   = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores  = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        center_angle = 0.0
        detected = False
        now = time.time()
        xmin = ymin = xmax = ymax = 0

        # --- Person detection check ---
        for i in range(len(scores)):
            if 0.59 < scores[i] <= 1.0:
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                object_name = labels[int(classes[i])]
                if object_name == 'person':
                    detected = True
                    last_person_box = (xmin, ymin, xmax, ymax)
                    last_confidence_time = now
                    break

        # --- Use last known position briefly ---
        if (not detected) and last_person_box and (now - last_confidence_time) < lost_timeout:
            xmin, ymin, xmax, ymax = last_person_box
            detected = True

        # --- Update person-visibility timers/state ---
        if detected:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            x = (xmin + xmax) / 2.0
            y = (ymin + ymax) / 2.0
            label = f"{int(x)},{int(y)}"
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame,
                          (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            person_visible = True
            last_seen_time = now
            if search_mode:
                search_mode = False
                search_stage = 0
        else:
            if (now - last_seen_time) > lost_timeout and not search_mode:
                search_mode = True
                search_stage = 0
                search_start = now
                scan_dir = 1 if last_seen_angle >= 0 else -1
        # Compute aim angle (either current detected center or last seen)
        frame_center = (imW / 2.0) + CENTER_BIAS_PX   # bias to fix camera offset
        center_angle_temp = 0.0
        if detected:
            x_center = (xmin + xmax) / 2.0
            x_offset = (x_center - frame_center) / frame_center  # [-1..1]
            center_angle_temp = x_offset * (CENTER_FOV / 2.0)
            last_seen_angle = center_angle_temp  # remember for search
        aim_angle = center_angle_temp if detected else last_seen_angle

        # Always refresh LiDAR around the aim angle
        read_lidar(aim_angle)
        center_angle = aim_angle  # keep this name alive for your (commented) plot block

        # --- Targets (PWM space) ---
        target_forward = 0.0
        target_turn = 0.0

        # =====================================================================
        # NORMAL FOLLOW (person visible and NOT searching)
        # =====================================================================

        if robot_enabled:
            print("on")
            if detected and not search_mode:
                # smooth x tracking
                smooth_x = x if smooth_x is None else (alpha * x + (1 - alpha) * smooth_x)

                frame_center = (imW / 2.0) + CENTER_BIAS_PX
                error_ratio = (smooth_x - frame_center) / frame_center  # [-1..1]

                # modest dead-zone: do nothing very near center
                if abs(error_ratio) < dead_zone_ratio:
                    target_turn = 0.0
                    prev_error = 0.0
                else:
                    # zero-cross short brake: if crossing near center, command zero
                    dt_ctrl = max(1e-3, now - last_ctrl_time)   # : dedicated controller dt
                    last_ctrl_time = now

                    if (np.sign(error_ratio) != np.sign(prev_error)) and (abs(error_ratio) < zero_cross_band):
                        target_turn = 0.0
                    else:
                        # PD turn control (damped) + slight nonlinearity: less gain near center
                        d_error = (error_ratio - prev_error) / dt_ctrl
                        d_error = float(np.clip(d_error, -6.0, 6.0))   # clamp derivative
                        prev_error = error_ratio
                        near_center_scale = 0.6 + 0.4 * abs(error_ratio)  # 0.6..1.0  
                        kP = turn_sensitivity * near_center_scale
                        target_turn = np.clip(
                            -turn * (kP * error_ratio + turn_D * d_error),
                            -turn, turn
                        )

                # follow distance band
                if front_distance is not None:
                    last_seen_dist = front_distance  # remember
                    if front_distance > FOLLOW_FAR:
                        target_forward = forward_speed
                    elif front_distance < FOLLOW_NEAR:
                        target_forward = -forward_speed
                    else:
                        target_forward = 0.0

                # obstacle shaping
                avoid_turn, allow_forward = 0.0, 1.0
                if left_min  and left_min  < SIDE_WARN: avoid_turn += 0.5
                if right_min and right_min < SIDE_WARN: avoid_turn -= 0.5
                if (left_min and left_min < SIDE_STOP) or (right_min and right_min < SIDE_STOP):
                    allow_forward = 0.0

                target_turn    += avoid_turn * turn
                target_turn    += TURN_BIAS                      # static turn bias
                target_forward *= allow_forward

                # predictive braking
                if (prev_dist is not None) and (front_distance is not None):
                    rate = (prev_dist - front_distance) / max(1e-3, (now - last_update))
                    if rate > CLOSE_RATE:
                        target_forward *= max(0.0, 1.0 - rate)
                prev_dist = front_distance

            elif search_mode:
                elapsed = now - search_start
                if search_stage == 0:
                    # stop & wait briefly
                    target_forward = 0.0
                    target_turn = 0.0
                    if elapsed >= STOP_PAUSE_SEC:
                        search_stage = 1
                        search_start = now
                elif search_stage == 1:
                    # slow 360 spin; flip direction periodically
                    target_forward = 0.0
                    target_turn = scan_dir * SEARCH_SPIN_TURN
                    if elapsed >= SEARCH_FLIP_SEC:
                        scan_dir *= -1
                        search_start = now

            dt_loop = now - last_update
            last_update = now

            ramp_step_fwd       = accel_rate_fwd        * dt_loop
            ramp_step_turn_up   = accel_rate_turn_up    * dt_loop
            ramp_step_turn_down = accel_rate_turn_down  * dt_loop

            def ramp(current, target, step):
                if current < target: return min(current + step, target)
                if current > target: return max(current - step, target)
                return current

            def ramp_asym_turn(current, target, step_up, step_down):
                # If magnitude is increasing on same sign → use step_up
                same_sign = (np.sign(current) == np.sign(target)) or (current == 0) or (target == 0)
                if same_sign and abs(target) > abs(current):
                    step = step_up
                else:
                    # braking or sign change → step_down (faster)
                    step = step_down
                if current < target: return min(current + step, target)
                if current > target: return max(current - step, target)
                return current

            current_forward = ramp(current_forward, target_forward, ramp_step_fwd)
            current_turn    = ramp_asym_turn(current_turn, target_turn, ramp_step_turn_up, ramp_step_turn_down)

            left  = np.clip(current_forward - current_turn, -100, 100)
            right = np.clip(current_forward + current_turn, -100, 100)

            left  = np.clip(left  * LEFT_GAIN,  -100, 100)
            right = np.clip(right * RIGHT_GAIN, -100, 100)

            tank(left, right)
            print(f"L:{left:.1f} R:{right:.1f} | mode={'SEARCH' if search_mode else 'FOLLOW'} | stage={search_stage if search_mode else '-'} | fd={front_distance} lm={left_min} rm={right_min}")

            cv2.putText(frame, f'FPS: {frame_rate_calc:.2f}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Object detector', frame)

            t2 = cv2.getTickCount()
            frame_rate_calc = 1 / ((t2 - t1) / freq)

            if laser.doProcessSimple(scan):
                xs, ys = [], []
                cone_width = FRONT_MARGIN
                fov_range = 3.0
                for p in scan.points:
                    if ((center_angle - cone_width) <= p.angle <= (center_angle + cone_width)
                            and RANGE_MIN < p.range <= fov_range):
                        xs.append(p.range * math.sin(p.angle))
                        ys.append(p.range * math.cos(p.angle))
                lidar_points_plot.set_data(xs, ys)
                fov_color = 'green' if detected else 'red'
                left_x = [0, fov_range * math.sin(center_angle - cone_width)]
                left_y = [0, fov_range * math.cos(center_angle - cone_width)]
                right_x = [0, fov_range * math.sin(center_angle + cone_width)]
                right_y = [0, fov_range * math.cos(center_angle + cone_width)]
                cone_fill_x = [0,
                            fov_range * math.sin(center_angle - cone_width),
                            fov_range * math.sin(center_angle + cone_width)]
                cone_fill_y = [0,
                            fov_range * math.cos(center_angle - cone_width),
                            fov_range * math.cos(center_angle + cone_width)]
                fov_fill.set_xy(np.column_stack((cone_fill_x, cone_fill_y)))
                fov_fill.set_facecolor(fov_color)
                fov_left_line.set_data(left_x, left_y)
                fov_right_line.set_data(right_x, right_y)
                fov_left_line.set_color(fov_color)
                fov_right_line.set_color(fov_color)
                fig.canvas.draw()
                fig.canvas.flush_events()

            if cv2.waitKey(1) == ord('q'):
                break
        else:
            reset_motion_state()

finally:
    # Clean shutdown
    try:
        tank(0, 0)
        disable_all()
    except:
        pass
    try:
        laser.turnOff()
        laser.disconnecting()
    except:
        pass
    try:
        cv2.destroyAllWindows()
    except:
        pass
    try:
        videostream.stop()
    except:
        pass
