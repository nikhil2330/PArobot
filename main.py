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
import os
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

turn = 70
forward_speed = 60

last_update = time.time()
current_forward = 0.0
current_turn = 0.0

smooth_x = None
dead_zone = 60 

last_person_box = None
last_confidence_time = 0
lost_timeout = 1.0  

# --- Behavior tuning ---
no_detect_counter = 0
accel_rate = 35.0           # faster ramp-up
alpha = 0.3                 # faster smoothing
dead_zone_ratio = 0.2       # more reactive center zone
turn_sensitivity = 0.9      # stronger turn response

# --- Main control loop ---
while True:
    frame_count += 1
    t1 = cv2.getTickCount()

    frame1 = videostream.read()
    if frame1 is None:
        continue
    frame = frame1.copy()
    
    frame_resized = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    center_angle = 0
    a = False
    now = time.time()
    
    # --- Object detection ---
    for i in range(len(scores)):
        if 0.59 < scores[i] <= 1.0:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            object_name = labels[int(classes[i])]
            if object_name == 'person':
                a = True
                last_person_box = (xmin, ymin, xmax, ymax)
                last_confidence_time = now
                break

    if not a and last_person_box and (now - last_confidence_time) < lost_timeout:
        xmin, ymin, xmax, ymax = last_person_box
        a = True
        print("Using last known position")

    target_forward = 0.0
    target_turn = 0.0

    # --- If person detected ---
    if a:
        no_detect_counter = 0
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
        x = (xmin + xmax) / 2.0
        y = (ymin + ymax) / 2.0

        if smooth_x is None:
            smooth_x = x
        else:
            smooth_x = alpha * x + (1 - alpha) * smooth_x

        frame_center = imW / 2.0
        center_error = smooth_x - frame_center
        error_ratio = center_error / frame_center  # normalized -1..1

        if abs(error_ratio) > dead_zone_ratio:
            target_turn = np.clip(-turn * error_ratio * turn_sensitivity, -turn, turn)
        else:
            target_turn = 0.0

        # --- LIDAR distance check ---
        fov = math.radians(60.0)
        x_offset = (x - frame_center) / frame_center
        center_angle = x_offset * (fov / 2.0)
        margin = math.radians(6.0)

        if laser.doProcessSimple(scan):
            angle_min = center_angle - margin
            angle_max = center_angle + margin
            target_points = [
                p.range for p in scan.points
                if (angle_min <= p.angle <= angle_max) and (0.15 < p.range)
            ]
            front_distance = np.median(sorted(target_points)[:3]) if target_points else None
        else:
            front_distance = None

        if front_distance is not None:
            print(f"Distance: {front_distance:.2f} m")
            if front_distance > 1.3:
                target_forward = forward_speed
            elif front_distance < 0.8:
                target_forward = -forward_speed
            else:
                target_forward = 0.0

        # --- Label ---
        label = str(int(x)) + "," + str(int(y))
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_ymin = max(ymin, labelSize[1] + 10)
        cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                      (xmin + labelSize[0], label_ymin + baseLine - 10),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        no_detect_counter += 1
        if no_detect_counter > 5:
            target_forward = 0.0
            target_turn = 0.0

    # --- Smooth acceleration ---
    dt = now - last_update
    last_update = now
    ramp_step = accel_rate * dt

    def ramp(current, target, step):
        if current < target:
            current = min(current + step, target)
        elif current > target:
            current = max(current - step, target)
        return current

    current_forward = ramp(current_forward, target_forward, ramp_step)
    current_turn = ramp(current_turn, target_turn, ramp_step)

    left = np.clip(current_forward - current_turn, -100, 100)
    right = np.clip(current_forward + current_turn, -100, 100)
    tank(left, right)
    print(f"L:{left:.1f} R:{right:.1f}")

    # --- Display ---
    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc),
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Object detector', frame)

    # --- FPS calc ---
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # --- LiDAR plot ---
    if laser.doProcessSimple(scan):
        xs, ys = [], []
        cone_width = math.radians(6.0)
        fov_range = 3.0
        for p in scan.points:
            if ((center_angle - cone_width) <= p.angle <= (center_angle + cone_width)
                    and 0.15 < p.range <= fov_range):
                xs.append(p.range * math.sin(p.angle))
                ys.append(p.range * math.cos(p.angle))

        lidar_points_plot.set_data(xs, ys)
        fov_color = 'green' if a else 'red'
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



