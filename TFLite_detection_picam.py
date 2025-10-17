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

lidar_points_plot, = ax.plot([], [], 'b.', markersize=4)
robot_dot, = ax.plot(0, 0, 'ro', markersize=6)

turn = 70
forward= 0.25

target_dist = 0.8  
min_dist = 0.4       
max_dist = 1.5       

correction = 0.2      
prev_forward = 0
prev_turn = 0
 
 

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    frame_count += 1
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()
    if frame1 is None:
        continue  # skip until a frame is ready
    frame = frame1.copy()
    
    
    frame_resized = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
    


    # Loop over all detections and draw detection box if confidence is above minimum threshold
    
    a = False
    for i in range(len(scores)):
        if ((scores[i] > 0.59) and (scores[i] <= 1.0)):
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            if object_name=='person':
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()

                a = True

                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                x = (xmin+xmax)/2
                y = (ymin + ymax)/2
                boxw = xmax - xmin
                boxh = ymax - ymin
                
                center = 100
                frame_center = imW / 2
                x_offset = (bbox_center_x - frame_center_x) / frame_center_x

                if x < frame_center - center:
                    l = -40
                    r = 40
                elif x > frame_center + center:
                    l = 40
                    r = -40
                else:
                    l = 0
                    r = 0
                    
                # ~ target_width = imW * 0.6   
                # ~ target_height = imH * 0.8 
                
                # ~ size = max(boxw/target_width, boxh/target_height)
                # ~ print(size)
                
                # ~ if size < 1.0:
                    # ~ forward = 50
                    # ~ print("forward")
                # ~ elif size > 1.2:
                    # ~ forward = -50
                    # ~ print("backward")
                # ~ else:
                    # ~ forward = 0

                # ~ tank(forward + l, forward + r)
                
                
                if laser.doProcessSimple(scan):
                    target_points = [p.range for p in scan.points if -0.1 <= p.angle <= 0.1 and p.range > 0.15]
                    if target_points: 
                        front_distance = np.median(sorted(target_points)[:3])
                        print(front_distance)
                        if front_distance >0.8:
                            forward = 50 
                            print("forward")  
                        elif front_distance < 0.6:
                            forward = -50 
                            print("backward")  
 
                        else:
                            forward = 0    

                        tank(forward + l, forward + r)

                    else:
                        tank(0, 0)  

                else:
                    tank(0, 0)  
                
                

                
                
                # label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                label = str(x) + " " + str(y) 
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                break
    if not a:
        tank(0, 0)
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    
    
    if laser.doProcessSimple(scan):
        xs, ys = [], []
        for p in scan.points:
            if -0.3 <= p.angle <= 0.3 and 0.15 < p.range <= 4.0: 
                xs.append(p.range * math.sin(p.angle))
                ys.append(p.range * math.cos(p.angle))
        lidar_points_plot.set_data(xs, ys)
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()

