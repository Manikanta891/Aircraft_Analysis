from flask import Flask, Response, jsonify, render_template
import cv2
import numpy as np
import json
import os
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time
import threading

config_w = None
config_h = None

script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(script_dir).resolve().parent
MODEL_PATH = ROOT_DIR / "Models" / "aircraft_detector_v11.pt"
VIDEO_PATH = ROOT_DIR / "Simulation_Videos" / "parking_simulation.mp4"
SLOTS_CONFIG_PATH = ROOT_DIR / "Airport_Simulator" / "parking.json"

app = Flask(__name__, template_folder='templates', static_folder='static')

TRAJECTORY_HISTORY_SIZE = 5
SPEED_THRESHOLD = 3.0
WARNING_DURATION = 60
CRITICAL_DURATION = 120

model = None
tracker = None
cap = None
slots = []
fps = 30

detector = None
frame_lock = threading.Lock()
current_frame = None
current_data = {}
is_running = False
processing_thread = None
video_loop_count = 0


@dataclass
class ParkingSlot:
    id: str
    x: int
    y: int
    w: int
    h: int
    aircraft_id: Optional[int] = None
    entry_time: Optional[float] = None
    status: str = "FREE"

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def get_duration(self) -> float:
        if self.entry_time is None:
            return 0.0
        return time.time() - self.entry_time


class ParkingMonitor:
    def __init__(self, slots_list: List[Dict]):
        self.slots: List[ParkingSlot] = []
        for i, box in enumerate(slots_list):
            slot_id = f"P{i+1}"
            self.slots.append(ParkingSlot(
                id=slot_id,
                x=box[0],
                y=box[1],
                w=box[2],
                h=box[3]
            ))
        self.event_logs: List[Dict] = []
        self.past_logs: List[Dict] = []
        self.long_stay_logged: Dict[str, bool] = {}
        self.critical_logged: Dict[str, bool] = {}

    def reset(self):
        for slot in self.slots:
            slot.aircraft_id = None
            slot.entry_time = None
            slot.status = "FREE"
        self.event_logs.clear()
        self.past_logs.extend(self.event_logs)
        self.long_stay_logged.clear()
        self.critical_logged.clear()

    def add_log(self, message: str, severity: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.event_logs.append({
            "timestamp": timestamp,
            "message": message,
            "severity": severity
        })
        if len(self.event_logs) > 30:
            self.event_logs = self.event_logs[-30:]

    def assign_aircraft_to_slots(self, aircraft_positions: Dict[int, Tuple[float, float]]):
        for slot in self.slots:
            slot_aircraft_id = slot.aircraft_id
            
            for aid, (cx, cy) in aircraft_positions.items():
                if slot.contains_point(cx, cy):
                    if slot.aircraft_id is None:
                        slot.aircraft_id = aid
                        slot.entry_time = time.time()
                        slot.status = "OCCUPIED"
                        self.add_log(f"Aircraft {aid} parked at {slot.id}", "INFO")
                    elif slot.aircraft_id == aid:
                        pass
                    else:
                        slot.status = "ERROR"
                    break
            else:
                if slot.aircraft_id is not None:
                    departed_id = slot.aircraft_id
                    duration = slot.get_duration()
                    slot.aircraft_id = None
                    slot.entry_time = None
                    slot.status = "FREE"
                    self.add_log(f"Aircraft {departed_id} departed from {slot.id} ({duration:.1f}s)", "INFO")

    def analyze_slots(self):
        alerts = []
        
        for slot in self.slots:
            if slot.status == "ERROR":
                alerts.append({
                    "type": "ERROR",
                    "message": f"Conflict in slot {slot.id}",
                    "slot_id": slot.id,
                    "severity_color": "red"
                })
                continue
            
            if slot.aircraft_id is not None:
                duration = slot.get_duration()
                
                if duration > CRITICAL_DURATION:
                    key = f"{slot.id}_{slot.aircraft_id}_critical"
                    if not self.critical_logged.get(key, False):
                        self.critical_logged[key] = True
                        self.add_log(f"Aircraft {slot.aircraft_id} at {slot.id} exceeded {CRITICAL_DURATION}s - CRITICAL", "CRITICAL")
                    alerts.append({
                        "type": "CRITICAL",
                        "message": f"Aircraft {slot.aircraft_id} at {slot.id} - {int(duration)}s",
                        "slot_id": slot.id,
                        "duration": int(duration),
                        "severity_color": "red"
                    })
                elif duration > WARNING_DURATION:
                    key = f"{slot.id}_{slot.aircraft_id}_warning"
                    if not self.long_stay_logged.get(key, False):
                        self.long_stay_logged[key] = True
                        self.add_log(f"Aircraft {slot.aircraft_id} at {slot.id} staying {int(duration)}s", "WARNING")
                    alerts.append({
                        "type": "WARNING",
                        "message": f"Aircraft {slot.aircraft_id} at {slot.id} - {int(duration)}s",
                        "slot_id": slot.id,
                        "duration": int(duration),
                        "severity_color": "orange"
                    })
        
        return alerts

    def get_slot_data(self) -> List[Dict]:
        slot_data = []
        for slot in self.slots:
            data = {
                "id": slot.id,
                "status": slot.status,
                "aircraft_id": slot.aircraft_id,
                "duration": int(slot.get_duration()) if slot.aircraft_id else 0
            }
            slot_data.append(data)
        return slot_data


def draw_slots(frame: np.ndarray, slots_list: List[ParkingSlot], active_slots: set):
    for slot in slots_list:
        x1, y1 = int(slot.x), int(slot.y)
        x2, y2 = int(slot.x + slot.w), int(slot.y + slot.h)

        if slot.status == "ERROR":
            color = (0, 0, 255)
            thickness = 3
        elif slot.status == "OCCUPIED":
            color = (0, 0, 255)
            thickness = 2
        else:
            color = (0, 255, 0)
            thickness = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        label = slot.id
        if slot.aircraft_id is not None:
            label = f"{slot.id}:ID{slot.aircraft_id}"
        
        cv2.putText(frame, label, (x1 + 5, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        duration = slot.get_duration()
        if slot.aircraft_id is not None and duration > 0:
            cv2.putText(frame, f"{int(duration)}s", (x1 + 5, y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def process_video():
    global current_frame, current_data, cap, detector, model, tracker, is_running, video_loop_count

    frame_count = 0
    prev_results = None

    while is_running:
        ret, frame = cap.read()

        if not ret:
            video_loop_count += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            detector.reset()
            tracker = DeepSort(max_age=30, n_init=3)
            prev_results = None
            print(f"Video looped - Reset (loop #{video_loop_count})")
            continue

        # 🔥 ONLY resize AFTER checking ret
        if config_w is not None and config_h is not None:
            frame = cv2.resize(frame, (config_w, config_h))

        frame_count += 1
        current_detections = []

        if frame_count % 2 == 0 or prev_results is None:
            results = model(frame, conf=0.25, imgsz=640)[0]
            prev_results = results
        else:
            results = prev_results

        if results.boxes is not None:
            for r in results.boxes.data:
                x1, y1, x2, y2, conf, cls = r.tolist()
                w = x2 - x1
                h = y2 - y1
                current_detections.append(([x1, y1, w, h], conf, 'aircraft'))

        tracks = tracker.update_tracks(current_detections, frame=frame)

        confirmed_tracks = []
        active_ids = set()
        aircraft_positions = {}

        for track in tracks:
            if not track.is_confirmed() or track.hits < 3:
                continue

            l, t, r, b = track.to_ltrb()
            cx = (l + r) / 2
            cy = (t + b) / 2
            track_id = track.track_id
            active_ids.add(track_id)

            aircraft_positions[track_id] = (cx, cy)

            confirmed_tracks.append({
                'id': track_id,
                'cx': cx,
                'cy': cy,
                'l': l, 't': t, 'r': r, 'b': b
            })

        detector.assign_aircraft_to_slots(aircraft_positions)
        alerts = detector.analyze_slots()

        active_slots = {i for i, slot in enumerate(detector.slots) if slot.status == "OCCUPIED"}
        draw_slots(frame, detector.slots, active_slots)

        for track in confirmed_tracks:
            tid = track['id']
            x1, y1, x2, y2 = int(track['l']), int(track['t']), int(track['r']), int(track['b'])
            cx, cy = int(track['cx']), int(track['cy'])

            slot_assigned = None
            for slot in detector.slots:
                if slot.aircraft_id == tid:
                    slot_assigned = slot
                    break

            if slot_assigned:
                color = (0, 0, 255)
                thickness = 3
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, f"ID:{tid} @ {slot_assigned.id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                color = (100, 100, 100)
                thickness = 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, f"ID:{tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        slot_data = detector.get_slot_data()
        occupied_count = sum(1 for s in slot_data if s['status'] == 'OCCUPIED')
        total_slots = len(slot_data)

        aircraft_in_slots = []
        for slot in detector.slots:
            if slot.aircraft_id is not None:
                aircraft_in_slots.append({
                    "id": slot.aircraft_id,
                    "slot": slot.id,
                    "duration": int(slot.get_duration())
                })

        past_logs = detector.past_logs[-10:] if detector.past_logs else []
        current_logs = detector.event_logs[-15:] if detector.event_logs else []
        all_logs = past_logs + current_logs

        data = {
            "parking": {
                "total_slots": total_slots,
                "occupied": occupied_count,
                "available": total_slots - occupied_count,
                "slots": slot_data,
                "aircraft": aircraft_in_slots,
                "alerts": alerts,
                "logs": all_logs[-20:] if len(all_logs) > 20 else all_logs
            },
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "video_loop": video_loop_count,
            "frame_count": frame_count
        }

        with frame_lock:
            _, buffer = cv2.imencode('.jpg', frame)
            current_frame = buffer.tobytes()
            current_data = data

        time.sleep(1.0 / fps)


def generate_frames():
    global current_frame
    while is_running:
        with frame_lock:
            if current_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
        time.sleep(0.03)

def initialize_system():
    global model, tracker, cap, slots, detector, is_running, processing_thread

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return False

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        return False

    if not os.path.exists(SLOTS_CONFIG_PATH):
        print(f"Error: Slots config not found at {SLOTS_CONFIG_PATH}")
        return False

    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    print("Initializing DeepSort tracker...")
    tracker = DeepSort(max_age=30, n_init=3)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {VIDEO_PATH}")
        return False

    # 🔥 Read one frame to get video size
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video frame")
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    with open(SLOTS_CONFIG_PATH, 'r') as f:
        config = json.load(f)
        slots = config.get('terminal_boxes', [])

    print(f"Slots loaded: {len(slots)}")

    # 🔥 Initialize detector
    detector = ParkingMonitor(slots)

    # 🔥 Start processing thread
    is_running = True
    processing_thread = threading.Thread(target=process_video)
    processing_thread.daemon = True
    processing_thread.start()

    print("✅ System initialized and running\n")
    return True


@app.route('/')
def index():
    return render_template('parking.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/data')
def get_data():
    with frame_lock:
        if not current_data:
            return jsonify({
                "parking": {
                    "total_slots": 0,
                    "occupied": 0,
                    "available": 0,
                    "slots": [],
                    "aircraft": [],
                    "alerts": [],
                    "logs": []
                },
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "video_loop": 0,
                "frame_count": 0
            })
        return jsonify(current_data)


@app.route('/api/start', methods=['POST'])
def start_stream():
    global is_running, processing_thread
    if not is_running:
        success = initialize_system()
        if success:
            return jsonify({"status": "started"})
        return jsonify({"status": "error", "message": "Failed to initialize"})
    return jsonify({"status": "already_running"})


@app.route('/api/stop', methods=['POST'])
def stop_stream():
    global is_running
    is_running = False
    if cap:
        cap.release()
    return jsonify({"status": "stopped"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)
