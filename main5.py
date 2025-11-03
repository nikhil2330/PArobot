######## SIMPLE PERSON FOLLOWING + LIDAR VISUALIZATION #########
# Keeps all original plotting, angle, LiDAR, and TensorFlow logic
# but uses simple tank movement (no PD, no ramp, no search, no obstacle avoid)

import os, argparse, cv2, numpy as np, time, math
from threading import Thread
import importlib.util
import ydlidar
import matplotlib.pyplot as plt
from gpiozero import LED, PWMLED
from picamera2 import Picamera2

# ----------------- BTS7960 PIN MAP (BOARD numbering) -----------------
l_rpwm = PWMLED("BOARD33", frequency=100)
l_lpwm = PWMLED("BOARD31", frequency=100)
l_en   = LED("BOARD29")

r_rpwm = PWMLED("BOARD35", frequency=100)
r_lpwm = PWMLED("BOARD37", frequency=100)
r_en   = LED("BOARD32")

INV_LEFT  = False
INV_RIGHT = False
MIN_PWM = 0.32

def enable_all():
    l_en.on(); r_en.on()

def disable_all():
    l_en.off(); r_en.off()

def _apply_deadband(val):
    if val == 0.0: return 0.0
    s = 1 if val > 0 else -1
    a = abs(val)
    return s * (MIN_PWM + (1.0 - MIN_PWM) * a)

def _drive_side(rpwm, lpwm, val, invert=False):
    if invert:
        val = -val
    if val > 0:
        lpwm.value, rpwm.value = 0, val
    elif val < 0:
        rpwm.value, lpwm.value = -val, 0
    else:
        lpwm.value = rpwm.value = 0

def tank(l, r):
    l = max(-100.0, min(100.0, float(l))) / 100.0
    r = max(-100.0, min(100.0, float(r))) / 100.0
    if INV_LEFT:  l = -l
    if INV_RIGHT: r = -r
    print(f"TANK raw L={l:.2f}, R={r:.2f}")  
    l = _apply_deadband(l)
    r = _apply_deadband(r)
    _drive_side(l_rpwm, l_lpwm, l)
    _drive_side(r_rpwm, r_lpwm, r, True)

enable_all()
print("Forward");  tank(40, 40); time.sleep(2)
print("Backward"); tank(-40, -40); time.sleep(2)
print("Left");     tank(-40, 40); time.sleep(2)
print("Right");    tank(40, -40); time.sleep(2)
print("Stop");     tank(0, 0)
disable_all()

# ----------------- CAMERA STREAM -----------------
class VideoStream:
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
        self.thread = Thread(target=self.update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            self.frame = self.picam2.capture_array("main")

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.picam2.close()

# ----------------- LIDAR INIT -----------------
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
    print("Failed to start LiDAR scan")
    exit(1)
scan = ydlidar.LaserScan()

# ----------------- MODEL LOAD -----------------
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', required=True)
parser.add_argument('--graph', default='detect.tflite')
parser.add_argument('--labels', default='labelmap.txt')
parser.add_argument('--threshold', default=0.5)
parser.add_argument('--resolution', default='1280x720')
parser.add_argument('--edgetpu', action='store_true')
args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])

if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5
outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# ----------------- PARAMETERS -----------------
forward_speed = 38
turn_speed = 30
dead_zone_ratio = 0.18
FOLLOW_FAR = 0.9
FOLLOW_NEAR = 0.8
RANGE_MIN = 0.15
RANGE_MAX = 4.0
CENTER_FOV = math.radians(60.0)
FRONT_MARGIN = math.radians(6.0)

# --- Plot setup ---
# plt.ion()
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.set_xlim(-1, 1)
# ax.set_ylim(0, 3)
# ax.grid(True)
# lidar_points_plot, = ax.plot([], [], 'b.', markersize=3, label="LIDAR points")
# robot_dot, = ax.plot(0, 0, 'ro', markersize=6, label="Robot")
# fov_fill = ax.fill([], [], 'orange', alpha=0.2, label="Camera FOV")[0]
# fov_left_line, = ax.plot([], [], 'orange', linewidth=1.5)
# fov_right_line, = ax.plot([], [], 'orange', linewidth=1.5)
# ax.legend(loc="upper right")

# ----------------- MAIN LOOP -----------------
videostream = VideoStream((imW, imH), framerate=30).start()
time.sleep(1)
enable_all()
time.sleep(0.1)

frame_rate_calc = 1
freq = cv2.getTickFrequency()
last_seen_angle = 0.0
front_distance = None

try:
    while True:
        t1 = cv2.getTickCount()
        frame = videostream.read()
        if frame is None:
            continue

        frame_resized = cv2.resize(frame, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        detected = False
        xmin = ymin = xmax = ymax = 0

        for i in range(len(scores)):
            if scores[i] > 0.6 and labels[int(classes[i])] == 'person':
                ymin = int(boxes[i][0] * imH)
                xmin = int(boxes[i][1] * imW)
                ymax = int(boxes[i][2] * imH)
                xmax = int(boxes[i][3] * imW)
                detected = True
                break

        frame_center = imW / 2.0
        center_angle = last_seen_angle

        if detected:
            x_center = (xmin + xmax) / 2.0
            x_offset = (x_center - frame_center) / frame_center
            center_angle = x_offset * (CENTER_FOV / 2.0)
            last_seen_angle = center_angle

        # --- LiDAR Read + FOV plot ---
        if laser.doProcessSimple(scan):
            xs, ys = [], []
            cone_width = FRONT_MARGIN
            fov_range = 3.0
            for p in scan.points:
                if ((center_angle - cone_width) <= p.angle <= (center_angle + cone_width)
                        and RANGE_MIN < p.range <= fov_range):
                    xs.append(p.range * math.sin(p.angle))
                    ys.append(p.range * math.cos(p.angle))
            # lidar_points_plot.set_data(xs, ys)
            # fov_color = 'green' if detected else 'red'
            # left_x = [0, fov_range * math.sin(center_angle - cone_width)]
            # left_y = [0, fov_range * math.cos(center_angle - cone_width)]
            # right_x = [0, fov_range * math.sin(center_angle + cone_width)]
            # right_y = [0, fov_range * math.cos(center_angle + cone_width)]
            # cone_fill_x = [0,
            #                fov_range * math.sin(center_angle - cone_width),
            #                fov_range * math.sin(center_angle + cone_width)]
            # cone_fill_y = [0,
            #                fov_range * math.cos(center_angle - cone_width),
            #                fov_range * math.cos(center_angle + cone_width)]
            # fov_fill.set_xy(np.column_stack((cone_fill_x, cone_fill_y)))
            # fov_fill.set_facecolor(fov_color)
            # fov_left_line.set_data(left_x, left_y)
            # fov_right_line.set_data(right_x, right_y)
            # fov_left_line.set_color(fov_color)
            # fov_right_line.set_color(fov_color)
            # fig.canvas.draw()
            # fig.canvas.flush_events()

            # Compute median front distance
            front_points = [p.range for p in scan.points
                            if (center_angle - FRONT_MARGIN) <= p.angle <= (center_angle + FRONT_MARGIN)
                            and RANGE_MIN < p.range <= RANGE_MAX]
            front_distance = np.median(front_points) if front_points else None

        # --- SIMPLE MOVEMENT ---
        if detected:
            x_center = (xmin + xmax) / 2.0
            x_offset = (x_center - frame_center) / frame_center
            if abs(x_offset) < dead_zone_ratio:
                # Centered: move based on distance
                if front_distance and front_distance > FOLLOW_FAR:
                    tank(forward_speed, forward_speed); move = "FORWARD"
                elif front_distance and front_distance < FOLLOW_NEAR:
                    tank(-forward_speed, -forward_speed); move = "BACK"
                else:
                    tank(0, 0); move = "STOP"
            elif x_offset > 0:
                tank(turn_speed, -turn_speed); move = "RIGHT"
            else:
                tank(-turn_speed, turn_speed); move = "LEFT"
        else:
            tank(0, 0); move = "STOP"

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
        cv2.putText(frame, f"Move:{move} Dist:{front_distance:.2f}" if front_distance else f"Move:{move}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        frame_rate_calc = 1 / ((t2 - t1) / freq)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    tank(0, 0)
    disable_all()
    try:
        laser.turnOff()
        laser.disconnecting()
    except:
        pass
    videostream.stop()
    cv2.destroyAllWindows()
    print("Shutdown complete.")
