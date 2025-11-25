#!/usr/bin/env python3
"""
Simple person detection + one-shot lock-on tracking (no motors, no LiDAR).

Locking logic:
- First detected person becomes the "candidate".
- While candidate is visible, we measure continuous wall-clock time:
    visible_time = now - visible_start_time.
- If candidate disappears EVEN FOR ONE FRAME:
    -> visible_start_time = None
    -> visible_time = 0
- When visible_time >= LOCK_VISIBLE_TIME_SEC AND candidate is visible
  in that frame, we LOCK onto that person.
- Once locked, we never change identity until the program exits.
"""

import os
import time
import argparse
import importlib.util

import numpy as np
import cv2

from picamera2 import Picamera2
from threading import Thread

# ============================
# CONFIG / CONSTANTS
# ============================

DETECTION_THRESHOLD = 0.5   # minimum confidence for person detection

LOCK_VISIBLE_TIME_SEC = 2.0  # candidate must be visible continuously for this many seconds
MIN_HIST_FRAMES       = 5    # minimum frames used to build candidate histogram

COLOR_SIM_THRESH      = 0.5  # minimum histogram correlation to accept as same person
HIST_BINS             = 16   # H & S bins for HSV histogram
MAX_TRACK_LOST_FRAMES = 5  # max consecutive frames allowed to lose track



# ============================
# CAMERA STREAM (Picamera2)
# ============================

class VideoStream:
    """
    Picamera2 video stream in a background thread.
    """

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


# ============================
# TFLITE MODEL SETUP + DETECTION
# ============================

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

    # Load labels
    with open(path_to_label, "r") as f:
        labels_local = [line.strip() for line in f.readlines()]
    if labels_local and labels_local[0] == "???":
        labels_local.pop(0)
    labels[:] = labels_local

    # Choose interpreter
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
    if "StatefulPartitionedCall" in outname:   # TF2-style
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:                                      # TF1-style
        boxes_idx, classes_idx, scores_idx = 0, 1, 2


def detect_people(frame_rgb, imW, imH, threshold):
    """
    Run TFLite model on frame_rgb, return a list of detections:
      [{'bbox': (xmin, ymin, xmax, ymax), 'score': float}, ...]
    Only includes 'person' class.
    """
    if interpreter is None:
        return []

    # Resize to model input size
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


# ============================
# COLOR HISTOGRAM HELPERS
# ============================

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


# ============================
# MAIN LOOP
# ============================

def main():
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

    print("[INFO] Initializing TFLite model...")
    init_tflite(args.modeldir, args.graph, args.labels, use_tpu=args.edgetpu)

    print("[INFO] Starting Picamera2 stream...")
    videostream = VideoStream(resolution=(imW, imH), framerate=30)

    # Candidate + track state
    candidate_active          = False
    candidate_bbox            = None
    candidate_hist_accum      = None
    candidate_hist_count      = 0
    candidate_visible_time    = 0.0   # seconds of continuous visibility
    candidate_visible_start_t = None  # when continuous visibility started

    tracked_active         = False
    tracked_bbox           = None
    tracked_hist           = None
    tracked_score          = 0.0
    tracked_lost_frames    = 0  # just for display/debug

    frame_rate_calc = 0.0
    freq = cv2.getTickFrequency()

    print("[INFO] Ready.")
    print("       1) Start the script.")
    print("       2) Make sure ONLY you are in front of the robot at first.")
    print(f"       3) Stay in view for ~{LOCK_VISIBLE_TIME_SEC} seconds continuously to lock.")

    try:
        while True:
            t1 = cv2.getTickCount()
            now = time.time()

            frame = videostream.read()
            if frame is None:
                continue

            frame_rgb = frame  # Picamera2 gives RGB888

            # Run detection
            people = detect_people(frame_rgb, imW, imH, args.threshold)

            state_str = "NO_PERSON"
            candidate_visible_this_frame = False

            # ========= BEFORE LOCK: build candidate =========
            if not tracked_active:
                # Draw all detections (yellow) ONLY before lock
                for det in people:
                    x1, y1, x2, y2 = det["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

                if not candidate_active:
                    # No candidate yet: if we see someone, that becomes the candidate.
                    if people:
                        best = max(people, key=lambda d: d["score"])
                        candidate_bbox = best["bbox"]
                        candidate_hist_accum = None
                        candidate_hist_count = 0
                        candidate_visible_time = 0.0
                        candidate_visible_start_t = None  # not yet visible
                        candidate_active = True

                        print("[INFO] Candidate chosen. Stay in view to lock.")
                        state_str = "CANDIDATE_STARTED"
                    else:
                        state_str = "WAITING_FOR_PERSON"
                else:
                    # Candidate already chosen.
                    if people:
                        # Assumption: during locking there is only one person,
                        # so just use highest-score detection.
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
                        # Candidate not visible this frame:
                        # reset continuous visibility timer
                        candidate_visible_this_frame = False
                        candidate_visible_start_t = None
                        candidate_visible_time = 0.0
                        state_str = "CANDIDATE_NOT_VISIBLE"

                    # --- continuous visibility timing using wall clock ---
                    if candidate_visible_this_frame:
                        if candidate_visible_start_t is None:
                            # just became visible after a gap -> start new interval
                            candidate_visible_start_t = now
                            candidate_visible_time = 0.0
                        else:
                            candidate_visible_time = now - candidate_visible_start_t
                    # if not visible, we already reset above

                    # Draw candidate in blue
                    if candidate_visible_this_frame and candidate_bbox is not None:
                        x1, y1, x2, y2 = candidate_bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Lock ONLY if:
                    #  - we've accumulated enough continuous visible time
                    #  - we have enough histogram frames
                    #  - candidate is visible THIS frame
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

                        print("[INFO] LOCKED onto person. This identity will not change until program exits.")
                        state_str = "LOCKED"

                    # ========= AFTER LOCK: track same person forever (NO re-lock) =========
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
                    # We re-found the SAME person (by color hist)
                    tracked_bbox = best_match["bbox"]
                    tracked_score = best_match["score"]
                    tracked_lost_frames = 0
                    state_str = f"LOCKED (sim={best_sim:.2f})"
                else:
                    # Person not visible in this frame
                    tracked_lost_frames += 1

                    if tracked_lost_frames > MAX_TRACK_LOST_FRAMES:
                        # Stop using stale bbox; robot should stop.
                        # IMPORTANT: we DO NOT reset tracked_active.
                        tracked_bbox = None
                        state_str = f"LOCKED_LOST_STOP ({tracked_lost_frames})"
                    else:
                        # Short "grace" phase where we still use last bbox
                        state_str = f"LOCKED_LOST ({tracked_lost_frames})"

                # Draw locked person (only when we have a CURRENT match) in red
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


            # ========= FPS + TIME OVERLAYS =========
            t2 = cv2.getTickCount()
            dt_total = (t2 - t1) / freq
            frame_rate_calc = 1.0 / dt_total if dt_total > 0 else 0.0

            cv2.putText(
                frame,
                f"FPS: {frame_rate_calc:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            if not tracked_active and candidate_active:
                cv2.putText(
                    frame,
                    f"VISIBLE: {candidate_visible_time:.2f}s / {LOCK_VISIBLE_TIME_SEC:.2f}s",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            cv2.putText(
                frame,
                f"STATE: {state_str}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            cv2.imshow("Person Lock Simple", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print("[INFO] Shutting down...")
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
