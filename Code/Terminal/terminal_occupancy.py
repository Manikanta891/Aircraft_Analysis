import cv2
import json
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(script_dir).resolve().parent.parent
TERMINALS_JSON = ROOT_DIR / "Airport_Simulator" / "terminals.json"
MODEL_PATH = ROOT_DIR / "Models" / "aircraft_detector_v11.pt"
VIDEO_PATH = ROOT_DIR / "Simulation_Videos" / "simulation-4.mp4"

with open(TERMINALS_JSON, "r") as f:
    terminals_data = json.load(f)
terminal_boxes = terminals_data["terminal_boxes"]

model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(str(VIDEO_PATH))
video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def point_in_box(point, box):
    x, y = point
    bx, by, bw, bh = box
    return bx <= x <= bx + bw and by <= y <= by + bh

while True:
    ret, frame = cap.read()
    if not ret:
        break

    terminal_status = ["Free"] * len(terminal_boxes)

    results = model(frame, conf=0.1, imgsz=960)[0]

    if results.boxes is not None:
        for r in results.boxes.data:
            x1, y1, x2, y2, conf, cls = r.tolist()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            for i, box in enumerate(terminal_boxes):
                if point_in_box((cx, cy), box):
                    terminal_status[i] = "Occupied"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    for i, box in enumerate(terminal_boxes):
        x, y, w, h = box
        color = (0, 255, 0) if terminal_status[i] == "Free" else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"G{i+1}", (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    panel_width = 200
    panel_height = 30 + len(terminal_boxes) * 25
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, "TERMINALS", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for i, status in enumerate(terminal_status):
        color = (0, 255, 0) if status == "Free" else (0, 165, 255)
        cv2.putText(frame, f"G{i+1}: {status}", (10, 45 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    display = cv2.resize(frame, (1200, 700))
    cv2.imshow("Terminal Gates", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()