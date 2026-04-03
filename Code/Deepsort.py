import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import csv


# -----------------------------
# PATH SETUP
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "Models" / "aircraft_detector_v8.pt"
VIDEO_PATH = ROOT_DIR / "Simulation_Videos" / "simulation-3.mp4"

# -----------------------------
# LOAD MODEL & TRACKER
# -----------------------------
model = YOLO(str(MODEL_PATH))
tracker = DeepSort(max_age=10, n_init=3)

cap = cv2.VideoCapture(str(VIDEO_PATH))

# -----------------------------
# LOG STORAGE
# -----------------------------
aircraft_log = {}
EXIT_THRESHOLD = 3  # seconds (tune this)

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # -----------------------------
    # YOLO DETECTION
    # -----------------------------
    results = model(frame, conf=0.15, imgsz=960)[0]

    detections = []

    if results.boxes is not None:
        for r in results.boxes.data:
            x1, y1, x2, y2, conf, cls = r.tolist()

            w = x2 - x1
            h = y2 - y1

            detections.append(([x1, y1, w, h], conf, 'aircraft'))

            # Draw detection box (green)
            cv2.rectangle(frame,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 255, 0), 2)

    # -----------------------------
    # TRACKING
    # -----------------------------
    tracks = tracker.update_tracks(detections, frame=frame)

    active_ids = set()

    for track in tracks:
        if not track.is_confirmed() or track.hits < 3:
            continue

        track_id = track.track_id
        active_ids.add(track_id)

        # Correct bounding box
        l, t, r, b = track.to_ltrb()

        # Center
        cx = int((l + r) / 2)
        cy = int((t + b) / 2)

        # Draw tracking box (blue)
        cv2.rectangle(frame,
                      (int(l), int(t)),
                      (int(r), int(b)),
                      (255, 0, 0), 2)

        # Centered ID
        text = f"ID: {track_id}"
        (tw, th), _ = cv2.getTextSize(text,
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6, 2)

        cv2.putText(frame,
                    text,
                    (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2)

        # -----------------------------
        # LOGIC: ENTRY + UPDATE
        # -----------------------------
        if track_id not in aircraft_log:
            aircraft_log[track_id] = {
                "entry_time": current_time,
                "last_seen": current_time,
                "exit_time": None,
                "duration": 0,
                "active": True
            }

        aircraft_log[track_id]["last_seen"] = current_time
        aircraft_log[track_id]["active"] = True

    # -----------------------------
    # CHECK EXITED AIRCRAFT
    # -----------------------------
    for tid, data in aircraft_log.items():
        if tid not in active_ids and data["active"]:
            if current_time - data["last_seen"] > EXIT_THRESHOLD:
                data["exit_time"] = current_time
                data["duration"] = data["exit_time"] - data["entry_time"]
                data["active"] = False

    # -----------------------------
    # DISPLAY COUNT
    # -----------------------------
    total_now = len(active_ids)

    cv2.putText(frame,
                f"Aircraft Now: {total_now}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2)

    # -----------------------------
    # DISPLAY FRAME
    # -----------------------------
    display = cv2.resize(frame, (1200, 700))
    cv2.imshow("Aircraft Monitoring System", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()

# -----------------------------
# EXPORT LOG TO CSV
# -----------------------------
with open("aircraft_log.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Track ID",
        "Entry Time",
        "Exit Time",
        "Duration (sec)",
        "Status"
    ])

    for tid, data in aircraft_log.items():
        entry = time.strftime('%H:%M:%S', time.localtime(data["entry_time"]))

        exit_t = (
            time.strftime('%H:%M:%S', time.localtime(data["exit_time"]))
            if data["exit_time"]
            else "Still Present"
        )

        duration = round(data["duration"], 2) if data["duration"] else 0

        status = "Active" if data["active"] else "Exited"

        writer.writerow([tid, entry, exit_t, duration, status])

print("✅ Log saved as aircraft_log.csv")