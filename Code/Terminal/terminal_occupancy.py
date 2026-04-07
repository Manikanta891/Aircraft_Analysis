import cv2
import json
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# LOAD TERMINAL DATA
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
terminals_path = os.path.join(script_dir, "terminals.json")

with open(terminals_path, "r") as f:
    terminals_data = json.load(f)

terminal_boxes_orig = terminals_data["terminal_boxes"]
orig_w = terminals_data["image_width"]
orig_h = terminals_data["image_height"]

# -----------------------------
# PATHS
# -----------------------------
ROOT_DIR = Path(script_dir).resolve().parent
MODEL_PATH = ROOT_DIR / "Models" / "aircraft_detector_v8.pt"
VIDEO_PATH = ROOT_DIR / "Simulation_Videos" / "simulation-4.mp4"

model = YOLO(str(MODEL_PATH))

# -----------------------------
# HELPERS
# -----------------------------
def point_in_box(point, box):
    x, y = point
    bx, by, bw, bh = box
    return bx <= x <= bx + bw and by <= y <= by + bh

def scale_boxes(boxes, ow, oh, vw, vh):
    sx = vw / ow
    sy = vh / oh
    scaled = []
    for x, y, w, h in boxes:
        scaled.append((
            int(x * sx),
            int(y * sy),
            int(w * sx),
            int(h * sy)
        ))
    return scaled

# -----------------------------
# VIDEO
# -----------------------------
cap = cv2.VideoCapture(str(VIDEO_PATH))

video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

terminal_boxes = scale_boxes(
    terminal_boxes_orig, orig_w, orig_h, video_w, video_h
)

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    terminal_status = ["Empty"] * len(terminal_boxes)

    # -----------------------------
    # DETECTION
    # -----------------------------
    results = model(frame, conf=0.1, imgsz=960)[0]

    if results.boxes is not None:
        for r in results.boxes.data:
            x1, y1, x2, y2, conf, cls = r.tolist()

            # center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # terminal logic (no drawing)
            for i, box in enumerate(terminal_boxes):
                if point_in_box((cx, cy), box):
                    terminal_status[i] = "Occupied"

            # draw ONLY aircraft
            cv2.rectangle(frame,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 255, 0), 2)

    # -----------------------------
    # DRAW TERMINAL STATUS PANEL
    # -----------------------------
    panel_width = 200
    panel_height = 30 + len(terminal_boxes) * 25

    # background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # title
    cv2.putText(frame,
                "TERMINALS",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)

    # status text
    for i, status in enumerate(terminal_status):
        color = (0, 255, 0) if status == "Occupied" else (0, 0, 255)

        cv2.putText(frame,
                    f"T{i+1}: {status}",
                    (10, 45 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1)

    # -----------------------------
    # DISPLAY
    # -----------------------------
    display = cv2.resize(frame, (1200, 700))
    cv2.imshow("Aircraft Detection + Terminal Status", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()