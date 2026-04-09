from flask import Flask, Response, jsonify, render_template
import cv2
import json
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time
import threading

script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(script_dir).resolve().parent

MODEL_PATH = ROOT_DIR / "Models" / "aircraft_detector_v11.pt"
VIDEO_PATH = ROOT_DIR / "Simulation_Videos" / "terminal_simulation.mp4"

TERMINALS_CONFIG = ROOT_DIR / "Airport_Simulator" / "terminals.json"

app = Flask(__name__, template_folder='templates', static_folder='static')

WARNING_DURATION = 60
CRITICAL_DURATION = 120
MAX_LOGS = 20

model = None
tracker = None
cap = None
fps = 30

frame_lock = threading.Lock()
current_frame = None
current_data = {}
is_running = False
processing_thread = None
video_loop_count = 0


@dataclass
class TerminalGate:
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


class TerminalMonitor:
    def __init__(self):
        self.gates: List[TerminalGate] = []
        self.terminal_logs: List[Dict] = []
        self.long_stay_logged: Dict[str, bool] = {}

    def add_log(self, message: str, severity: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.terminal_logs.append({
            "timestamp": timestamp,
            "message": message,
            "severity": severity
        })
        if len(self.terminal_logs) > MAX_LOGS:
            self.terminal_logs = self.terminal_logs[-MAX_LOGS:]

    def reset(self):
        for gate in self.gates:
            gate.aircraft_id = None
            gate.entry_time = None
            gate.status = "FREE"
        self.terminal_logs.clear()
        self.long_stay_logged.clear()

    def update_aircraft_positions(self, aircraft_positions: Dict[int, Tuple[float, float]]):
        for aid, (cx, cy) in aircraft_positions.items():
            for gate in self.gates:
                if gate.contains_point(cx, cy):
                    if gate.aircraft_id is None:
                        gate.aircraft_id = aid
                        gate.entry_time = time.time()
                        gate.status = "OCCUPIED"
                        self.add_log(f"Aircraft {aid} docked at gate {gate.id}", "INFO")
                    elif gate.aircraft_id != aid:
                        gate.status = "ERROR"

        for gate in self.gates:
            if gate.aircraft_id is not None and gate.aircraft_id not in aircraft_positions:
                duration = gate.get_duration()
                self.add_log(f"Aircraft {gate.aircraft_id} departed gate {gate.id} ({int(duration)}s)", "INFO")
                gate.aircraft_id = None
                gate.entry_time = None
                gate.status = "FREE"

        self._check_long_stays()

    def _check_long_stays(self):
        for gate in self.gates:
            if gate.aircraft_id is not None:
                duration = gate.get_duration()
                key = f"terminal_{gate.id}_{gate.aircraft_id}"
                if duration > CRITICAL_DURATION:
                    if not self.long_stay_logged.get(f"{key}_critical", False):
                        self.long_stay_logged[f"{key}_critical"] = True
                        self.add_log(f"Aircraft {gate.aircraft_id} at gate {gate.id} exceeded {CRITICAL_DURATION}s", "CRITICAL")
                elif duration > WARNING_DURATION:
                    if not self.long_stay_logged.get(f"{key}_warning", False):
                        self.long_stay_logged[f"{key}_warning"] = True
                        self.add_log(f"Aircraft {gate.aircraft_id} staying at gate {gate.id} ({int(duration)}s)", "WARNING")

    def get_terminal_data(self) -> Dict:
        total_gates = len(self.gates)
        occupied = sum(1 for g in self.gates if g.status == "OCCUPIED")
        available = total_gates - occupied

        gates_data = []
        for gate in self.gates:
            gates_data.append({
                "id": gate.id,
                "status": gate.status,
                "aircraft_id": f"AC-{gate.aircraft_id}" if gate.aircraft_id else None,
                "duration": int(gate.get_duration()) if gate.aircraft_id else 0
            })

        terminal_alerts = []
        for gate in self.gates:
            if gate.aircraft_id is not None:
                duration = gate.get_duration()
                if duration > CRITICAL_DURATION:
                    terminal_alerts.append({
                        "type": "CRITICAL",
                        "message": f"Long stay: AC-{gate.aircraft_id} at {gate.id}",
                        "severity": "red"
                    })
                elif duration > WARNING_DURATION:
                    terminal_alerts.append({
                        "type": "WARNING",
                        "message": f"Extended stay: AC-{gate.aircraft_id} at {gate.id}",
                        "severity": "orange"
                    })

        aircraft_at_terminals = []
        for gate in self.gates:
            if gate.aircraft_id is not None:
                status = "DELAY" if gate.get_duration() > WARNING_DURATION else "DOCKED"
                aircraft_at_terminals.append({
                    "id": f"AC-{gate.aircraft_id}",
                    "gate": gate.id,
                    "status": status,
                    "duration": int(gate.get_duration())
                })

        overall_status = "SAFE"
        if any(a["type"] == "CRITICAL" for a in terminal_alerts):
            overall_status = "CRITICAL"
        elif any(a["type"] == "WARNING" for a in terminal_alerts):
            overall_status = "WARNING"

        return {
            "total_gates": total_gates,
            "occupied": occupied,
            "available": available,
            "gates": gates_data,
            "aircraft": aircraft_at_terminals,
            "alerts": terminal_alerts,
            "logs": self.terminal_logs[-MAX_LOGS:] if self.terminal_logs else [],
            "summary": {
                "total_aircraft": len(aircraft_at_terminals),
                "overall_status": overall_status,
                "critical_count": sum(1 for a in terminal_alerts if a["type"] == "CRITICAL"),
                "warning_count": sum(1 for a in terminal_alerts if a["type"] == "WARNING")
            }
        }


def load_configurations():
    global monitor

    monitor.gates.clear()

    if os.path.exists(TERMINALS_CONFIG):
        with open(TERMINALS_CONFIG, 'r') as f:
            term_data = json.load(f)
            boxes = term_data.get("terminal_boxes", [])

            for i, box in enumerate(boxes):
                monitor.gates.append(TerminalGate(
                    id=f"G{i+1}",
                    x=int(box[0]),
                    y=int(box[1]),
                    w=int(box[2]),
                    h=int(box[3])
                ))
        print(f"Loaded {len(monitor.gates)} gate(s)")
    else:
        print("Warning: terminals.json not found, using default gate positions")
        for i in range(4):
            monitor.gates.append(TerminalGate(
                id=f"G{i+1}",
                x=100 + (i % 2) * 200,
                y=300 + (i // 2) * 150,
                w=80,
                h=50
            ))


monitor = TerminalMonitor()


def process_video():
    global current_frame, current_data, cap, tracker, is_running, video_loop_count

    frame_count = 0
    prev_results = None

    while is_running:
        ret, frame = cap.read()
        if not ret:
            video_loop_count += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            tracker = DeepSort(max_age=30, n_init=3)
            monitor.reset()
            prev_results = None
            print(f"Video looped - Reset (loop #{video_loop_count})")
            continue

        frame_count += 1
        current_detections = []

        if frame_count % 2 == 0 or prev_results is None:
            results = model(frame, conf=0.2, imgsz=640)[0]
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

        aircraft_positions = {}
        for track in tracks:
            if not track.is_confirmed() or track.hits < 2:
                continue
            l, t, r, b = track.to_ltrb()
            cx = (l + r) / 2
            cy = (t + b) / 2
            track_id = track.track_id
            aircraft_positions[track_id] = (cx, cy)

        monitor.update_aircraft_positions(aircraft_positions)

        terminal_data = monitor.get_terminal_data()

        # -----------------------------
        # DRAW TERMINAL BOUNDING BOXES
        # -----------------------------
        for gate in monitor.gates:
            x, y, w, h = gate.x, gate.y, gate.w, gate.h

            # Color based on status
            if gate.status == "FREE":
                color = (0, 255, 0)        # Green
            elif gate.status == "OCCUPIED":
                color = (0, 165, 255)      # Orange
            # else:
            #     color = (0, 0, 255)        # Red (ERROR)

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Label text
            label = f"{gate.id}"
            if gate.aircraft_id:
                label += f" | AC-{gate.aircraft_id}"

            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - 20), (x + tw, y), color, -1)

            # Put text
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        data = {
            "terminal": terminal_data,
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
    global model, tracker, cap, is_running, processing_thread

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return False

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        return False

    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    print("Initializing DeepSort tracker...")
    tracker = DeepSort(max_age=30, n_init=3)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {VIDEO_PATH}")
        return False

    ret_test, frame_test = cap.read()
    if not ret_test:
        print(f"Error: Cannot read video frame")
        cap.release()
        return False

    video_h, video_w = frame_test.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"Video dimensions: {video_w}x{video_h}")
    load_configurations()

    is_running = True
    processing_thread = threading.Thread(target=process_video)
    processing_thread.daemon = True
    processing_thread.start()

    print("System initialized and running")
    return True


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/data')
def get_data():
    with frame_lock:
        if not current_data:
            return jsonify({
                "terminal": {
                    "total_gates": 0, "occupied": 0, "available": 0,
                    "gates": [], "aircraft": [], "alerts": [], "logs": [],
                    "summary": {"total_aircraft": 0, "overall_status": "SAFE", "critical_count": 0, "warning_count": 0}
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
    app.run(debug=True, host='0.0.0.0', port=5004)
