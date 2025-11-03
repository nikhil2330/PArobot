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




#song = AudioSegment.from_mp3("person.mp3")
l1 = LED("BOARD13")
l2 = LED("BOARD11")
le = PWMLED("BOARD18")
r1 = LED("BOARD22")
r2 = LED("BOARD40")
re = PWMLED("BOARD12")



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

def tank(l,r):
    l /= 100
    r /= 100
    if l<0:
        l1.off()
        l2.on()
        le.value = abs(l)
    elif l==0:
        l1.off()
        l2.off()
    else:
        l1.on()
        l2.off()
        le.value = abs(l)
    if r<0:
        r1.off()
        r2.on()
        re.value = abs(r)
    elif r==0:
        r1.off()
        r2.off()
    else:
        r1.on()
        r2.off()
        re.value = abs(r)


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
turn = 90                   # max turning PWM output
forward_speed = 60          # max forward PWM output

# --- Behavior tuning ---
accel_rate = 50.0           # rate of change for speed (units/sec)
alpha = 0.3                 # smoothing factor for person x-position (0-1)
dead_zone_ratio = 0.1       # ignore small horizontal errors (normalized -1..1)
turn_sensitivity = 0.9      # scale turning reactivity
lost_timeout = 1.0          # sec - how long to remember last seen person

# --- Distance thresholds (in meters) ---
FOLLOW_FAR   = 1.10         # go forward if person farther than this
FOLLOW_NEAR  = 0.90         # back up if closer than this
SIDE_WARN    = 0.60         # start turning away if obstacle nearer than this
SIDE_STOP    = 0.40         # stop forward motion if obstacle too close
RANGE_MIN    = 0.15         # ignore invalid or ultra-close lidar hits
RANGE_MAX    = 3.00         # ignore far lidar hits
CLOSE_RATE   = 0.30         # m/s - braking trigger (approach rate)

# --- LIDAR cone geometry (radians) ---
CENTER_FOV   = math.radians(60.0)   # total width of front tracking cone
FRONT_MARGIN = math.radians(6.0)    # half-angle for center cone
SIDE_OFFSET  = math.radians(20.0)   # offset from center to start side cone
SIDE_SPREAD  = math.radians(40.0)   # side cone angular spread

# --- Initialization for motion state ---
last_update = time.time()
current_forward = 0.0
current_turn = 0.0

prev_dist = None       # last distance to person (for braking)
front_distance = None  # current measured distance to person
left_min = None        # nearest obstacle left
right_min = None       # nearest obstacle right

smooth_x = None        # smoothed person x-center

last_person_box = None
last_confidence_time = 0


person_visible = False
last_seen_time = 0.0
last_seen_angle = 0.0
last_seen_dist = 1.0
recovery_mode = False
recovery_phase = 0
recovery_start = 0.0
SAFETY_MARGIN = 0.3
RECOVERY_TURN_SPEED = 0.6
LOST_TIMEOUT = 1.0


# --- MAIN CONTROL LOOP --------------------------------------------------------

# --- MAIN CONTROL LOOP --------------------------------------------------------
while True:
    frame_count += 1
    t1 = cv2.getTickCount()
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
        print("Using last known position")

    # --- Update person-visibility timers/state ---
    if detected:
        person_visible = True
        last_seen_time = now
    else:
        # trigger recovery once we've exceeded the timeout
        if (now - last_seen_time) > LOST_TIMEOUT and not recovery_mode:
            person_visible = False
            recovery_mode = True
            recovery_phase = 0
            recovery_start = now
            print("Person lost → starting recovery sequence")

    # --- Targets (PWM space) ---
    target_forward = 0.0
    target_turn = 0.0

    # =====================================================================
    # NORMAL FOLLOW (person visible / remembered this frame, not recovering)
    # =====================================================================
    if detected and not recovery_mode:
        no_detect_counter = 0

        # draw bbox + label
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

        # smooth x tracking
        smooth_x = x if smooth_x is None else (alpha * x + (1 - alpha) * smooth_x)

        frame_center = imW / 2.0
        center_error = smooth_x - frame_center
        error_ratio = center_error / frame_center  # [-1..1]

        # horizontal control
        if abs(error_ratio) > dead_zone_ratio:
            target_turn = np.clip(-turn * error_ratio * turn_sensitivity, -turn, turn)
        else:
            target_turn = 0.0

        # lidar angle aligned to person
        x_offset = (x - frame_center) / frame_center
        center_angle = x_offset * (CENTER_FOV / 2.0)
        last_seen_angle = center_angle  # remember for recovery

        # --- LiDAR sampling ---
        if laser.doProcessSimple(scan):
            # front (person-centered)
            a_min = center_angle - FRONT_MARGIN
            a_max = center_angle + FRONT_MARGIN
            target_points = [p.range for p in scan.points
                             if (a_min <= p.angle <= a_max) and (RANGE_MIN < p.range)]
            front_distance = np.median(sorted(target_points)[:3]) if target_points else None

            # sides (person-relative)
            left_points = [p.range for p in scan.points
                           if (center_angle + SIDE_OFFSET) < p.angle < (center_angle + SIDE_OFFSET + SIDE_SPREAD)
                           and RANGE_MIN < p.range < RANGE_MAX]
            right_points = [p.range for p in scan.points
                            if (center_angle - SIDE_OFFSET - SIDE_SPREAD) < p.angle < (center_angle - SIDE_OFFSET)
                            and RANGE_MIN < p.range < RANGE_MAX]
            left_min  = min(left_points)  if left_points  else None
            right_min = min(right_points) if right_points else None

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
            avoid_turn, avoid_forward = 0.0, 1.0
            if left_min  and left_min  < SIDE_WARN: avoid_turn += 0.5
            if right_min and right_min < SIDE_WARN: avoid_turn -= 0.5
            if (left_min and left_min < SIDE_STOP) or (right_min and right_min < SIDE_STOP):
                avoid_forward = 0.0

            target_turn    += avoid_turn * turn
            target_forward *= avoid_forward

            # predictive braking
            if (prev_dist is not None) and (front_distance is not None):
                rate = (prev_dist - front_distance) / (now - last_update)
                if rate > CLOSE_RATE:
                    target_forward *= max(0.0, 1.0 - rate)
            prev_dist = front_distance

    # =====================================================================
    # RECOVERY MODE
    # =====================================================================
    elif recovery_mode:
        # safe fallbacks when no fresh lidar
        f = front_distance if front_distance is not None else 99.0
        l = left_min if left_min is not None else 99.0
        r = right_min if right_min is not None else 99.0
        target_forward, target_turn = 0.0, 0.0

        if recovery_phase == 0:
            # (0) TURN AWAY until the front clears
            if f < 0.7:
                target_turn = RECOVERY_TURN_SPEED if l >= r else -RECOVERY_TURN_SPEED
            else:
                recovery_phase = 1
                recovery_start = now
                print("→ Phase 1 done: start forward move")

        elif recovery_phase == 1:
            # (1) MOVE FORWARD until near the last seen distance (minus margin)
            target_forward = forward_speed * 0.5
            if (front_distance is not None) and (front_distance <= (last_seen_dist - SAFETY_MARGIN)):
                recovery_phase = 2
                recovery_start = now
                print("→ Phase 2: peek around corner")
            elif (front_distance is not None) and (front_distance < 0.4):
                target_forward = 0.0  # hard stop if something too close

        elif recovery_phase == 2:
            # (2) PEEK: rotate toward last-seen angle; creep if clear
            target_turn = RECOVERY_TURN_SPEED if last_seen_angle >= 0 else -RECOVERY_TURN_SPEED
            if f > 1.0:
                target_forward = forward_speed * 0.4
            # exit recovery as soon as the person is seen
            if detected:
                recovery_mode = False
                recovery_phase = 0
                prev_dist = None
                person_visible = True
                last_seen_time = now
                print("Person reacquired → back to follow mode")

    # =====================================================================
    # SMOOTH ACCEL + DRIVE
    # =====================================================================
    dt = now - last_update
    last_update = now
    ramp_step = accel_rate * dt

    def ramp(current, target, step):
        if current < target: return min(current + step, target)
        if current > target: return max(current - step, target)
        return current

    current_forward = ramp(current_forward, target_forward, ramp_step)
    current_turn    = ramp(current_turn,    target_turn,    ramp_step)

    left  = np.clip(current_forward - current_turn, -100, 100)
    right = np.clip(current_forward + current_turn, -100, 100)
    tank(left, right)
    print(f"L:{left:.1f} R:{right:.1f} | mode={'RECOVERY' if recovery_mode else 'FOLLOW'} | phase={recovery_phase if recovery_mode else '-'}")

    # --- FPS + frame ---
    cv2.putText(frame, f'FPS: {frame_rate_calc:.2f}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Object detector', frame)

    # --- FPS calc ---
    t2 = cv2.getTickCount()
    frame_rate_calc = 1 / ((t2 - t1) / freq)

    # --- Visualization update (lidar FOV) ---
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

cv2.destroyAllWindows()
videostream.stop()
