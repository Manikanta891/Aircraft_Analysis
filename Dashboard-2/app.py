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

script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(script_dir).resolve().parent
MODEL_PATH = ROOT_DIR / "Models" / "aircraft_detector_v8.pt"
VIDEO_PATH = ROOT_DIR / "Simulation_Videos" / "restricted_video.mp4"
ZONES_CONFIG_PATH = ROOT_DIR / "Airport_Simulator" / "restricted_zones.json"

app = Flask(__name__, template_folder='templates', static_folder='static')

TRAJECTORY_HISTORY_SIZE = 8
MIN_TRAJECTORY_FRAMES = 5
SPEED_THRESHOLD = 3.0
WARNING_TIME = 5
CRITICAL_TIME = 10

model = None
tracker = None
cap = None
zones = []
fps = 30

detector = None
frame_lock = threading.Lock()
current_frame = None
current_data = {}
is_running = False
processing_thread = None
video_loop_count = 0
init_complete = False


@dataclass
class AircraftTrack:
    track_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_SIZE))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_SIZE))
    in_restricted_zone: bool = False
    zone_id: Optional[int] = None
    entry_time: Optional[float] = None
    has_entered: bool = False
    has_exited: bool = False
    last_exit_time: Optional[float] = None

    def get_current_position(self) -> Optional[Tuple[float, float]]:
        if len(self.positions) < 1:
            return None
        return self.positions[-1]

    def get_oldest_position(self) -> Optional[Tuple[float, float]]:
        if len(self.positions) < MIN_TRAJECTORY_FRAMES:
            return None
        return self.positions[0]

    def get_trajectory_vector(self) -> Optional[Tuple[float, float, float]]:
        current = self.get_current_position()
        oldest = self.get_oldest_position()
        if current is None or oldest is None:
            return None
        mvx = current[0] - oldest[0]
        mvy = current[1] - oldest[1]
        speed = np.sqrt(mvx**2 + mvy**2)
        return mvx, mvy, speed

    def get_average_speed(self) -> float:
        result = self.get_trajectory_vector()
        if result is None:
            return 0.0
        return result[2]

    def is_static(self) -> bool:
        return self.get_average_speed() < SPEED_THRESHOLD


class RestrictedZoneMonitor:
    def __init__(self, zones_list: List[Dict]):
        self.zones = zones_list
        self.tracks: Dict[int, AircraftTrack] = {}
        self.event_logs: List[Dict] = []
        self.past_logs: List[Dict] = []
        self.start_time = time.time()
        self.warning_logged: Dict[int, bool] = {}
        self.critical_logged: Dict[int, bool] = {}

    def reset(self):
        self.tracks.clear()
        self.event_logs.clear()
        self.past_logs.extend(self.event_logs)
        self.warning_logged.clear()
        self.critical_logged.clear()

    def is_point_in_box(self, px: float, py: float, box: dict) -> bool:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)

    def check_restricted_zone(self, cx: float, cy: float) -> Tuple[bool, Optional[Dict]]:
        for zone in self.zones:
            restricted = zone.get('restricted', zone)
            if self.is_point_in_box(cx, cy, restricted):
                return True, zone
        return False, None

    def add_log(self, message: str, severity: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.event_logs.append({
            "timestamp": timestamp,
            "message": message,
            "severity": severity
        })
        if len(self.event_logs) > 30:
            self.event_logs = self.event_logs[-30:]

    def update_track(self, track_id: int, cx: float, cy: float, frame_num: int):
        was_inside = False
        if track_id in self.tracks:
            was_inside = self.tracks[track_id].in_restricted_zone

        if track_id not in self.tracks:
            self.tracks[track_id] = AircraftTrack(track_id=track_id)
            self.tracks[track_id].entry_time = time.time()

        in_restricted, zone = self.check_restricted_zone(cx, cy)
        self.tracks[track_id].in_restricted_zone = in_restricted
        self.tracks[track_id].zone_id = zone['id'] if zone else None
        self.tracks[track_id].positions.append((cx, cy))
        self.tracks[track_id].timestamps.append(frame_num)

        if in_restricted and not was_inside:
            self.tracks[track_id].has_entered = True
            if self.tracks[track_id].has_exited:
                self.add_log(f"Aircraft {track_id} re-entered zone", "INFO")
            else:
                self.add_log(f"Aircraft {track_id} entered restricted zone", "INFO")
            self.tracks[track_id].entry_time = time.time()
            self.warning_logged[track_id] = False
            self.critical_logged[track_id] = False

        elif not in_restricted and was_inside:
            self.tracks[track_id].has_exited = True
            self.tracks[track_id].has_entered = False
            self.tracks[track_id].last_exit_time = time.time()
            self.add_log(f"Aircraft {track_id} exited restricted zone", "INFO")

    def get_aircraft_in_zone(self) -> List[AircraftTrack]:
        return [t for t in self.tracks.values() if t.in_restricted_zone]

    def analyze_severity(self, track: AircraftTrack) -> Tuple[str, float]:
        if not track.in_restricted_zone or track.entry_time is None:
            return "NORMAL", 0.0

        time_inside = time.time() - track.entry_time

        if time_inside > CRITICAL_TIME:
            return "CRITICAL", time_inside
        elif time_inside > WARNING_TIME:
            return "WARNING", time_inside
        else:
            return "NORMAL", time_inside

    def cleanup_tracks(self, active_ids: set):
        ids_to_remove = [tid for tid in self.tracks.keys() if tid not in active_ids]
        for tid in ids_to_remove:
            if tid in self.tracks:
                del self.tracks[tid]
            if tid in self.warning_logged:
                del self.warning_logged[tid]
            if tid in self.critical_logged:
                del self.critical_logged[tid]


def draw_zones(frame: np.ndarray, zones_list: List[Dict], active_zones: set):
    for zone in zones_list:
        restricted = zone.get('restricted', zone)
        r_x1, r_y1 = int(restricted['x1']), int(restricted['y1'])
        r_x2, r_y2 = int(restricted['x2']), int(restricted['y2'])

        is_active = zone['id'] in active_zones

        overlay = frame.copy()
        cv2.rectangle(overlay, (r_x1, r_y1), (r_x2, r_y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.rectangle(frame, (r_x1, r_y1), (r_x2, r_y2), (0, 0, 255), 3)

        label = zone.get('name', f"Zone {zone['id']}")
        cx = (r_x1 + r_x2) // 2
        cy = r_y1 - 10
        color = (0, 0, 255) if is_active else (255, 100, 100)
        cv2.putText(frame, label, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_trajectory_arrow(frame: np.ndarray, start: Tuple[int, int],
                         vector: Tuple[float, float, float],
                         color: Tuple[int, int, int], length_scale: float = 0.5):
    if vector is None:
        return
    mvx, mvy, speed = vector
    if speed < SPEED_THRESHOLD:
        return
    norm_mvx = mvx / speed * 50 * length_scale
    norm_mvy = mvy / speed * 50 * length_scale
    end_x = int(start[0] + norm_mvx)
    end_y = int(start[1] + norm_mvy)
    cv2.arrowedLine(frame, start, (end_x, end_y), color, 2, tipLength=0.3)


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

        for track in tracks:
            if not track.is_confirmed() or track.hits < 3:
                continue

            l, t, r, b = track.to_ltrb()
            cx = (l + r) / 2
            cy = (t + b) / 2
            track_id = track.track_id
            active_ids.add(track_id)

            detector.update_track(track_id, cx, cy, frame_count)

            track_data = detector.tracks.get(track_id)
            is_in_zone = track_data.in_restricted_zone if track_data else False

            confirmed_tracks.append({
                'id': track_id,
                'cx': cx,
                'cy': cy,
                'l': l, 't': t, 'r': r, 'b': b,
                'in_restricted': is_in_zone,
                'zone_id': track_data.zone_id if track_data else None,
                'track_data': track_data
            })

        detector.cleanup_tracks(active_ids)

        active_zones = set()
        restricted_aircraft = detector.get_aircraft_in_zone()
        for aircraft in restricted_aircraft:
            if aircraft.zone_id:
                active_zones.add(aircraft.zone_id)

        draw_zones(frame, zones, active_zones)

        for track in confirmed_tracks:
            tid = track['id']
            x1, y1, x2, y2 = int(track['l']), int(track['t']), int(track['r']), int(track['b'])
            cx, cy = int(track['cx']), int(track['cy'])
            track_data = track['track_data']
            trajectory = track_data.get_trajectory_vector() if track_data else None

            if track['in_restricted']:
                color = (0, 0, 255)
                thickness = 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, f"ID:{tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                zone_name = next((z['name'] for z in zones if z['id'] == track['zone_id']), "Zone")
                cv2.putText(frame, f"IN {zone_name.upper()}", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                color = (100, 100, 100)
                thickness = 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, f"ID:{tid}", (cx - 15, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                if trajectory and track_data and not track_data.is_static():
                    draw_trajectory_arrow(frame, (cx, cy), trajectory, (0, 255, 0), 0.5)

        for aircraft in restricted_aircraft:
            severity, time_inside = detector.analyze_severity(aircraft)

            if severity == "CRITICAL" and not detector.critical_logged.get(aircraft.track_id, False):
                detector.critical_logged[aircraft.track_id] = True
                detector.add_log(f"Aircraft {aircraft.track_id} exceeded {CRITICAL_TIME}s limit - CRITICAL", "CRITICAL")
            elif severity == "WARNING" and not detector.warning_logged.get(aircraft.track_id, False):
                detector.warning_logged[aircraft.track_id] = True
                detector.add_log(f"Aircraft {aircraft.track_id} staying {WARNING_TIME}+ seconds - WARNING", "WARNING")

        aircraft_count = len(restricted_aircraft)

        if aircraft_count >= 2:
            zone_status = "VIOLATED"
        elif aircraft_count == 1:
            severity, _ = detector.analyze_severity(restricted_aircraft[0])
            zone_status = "VIOLATED" if severity == "CRITICAL" else "ACTIVE"
        else:
            zone_status = "CLEAR"

        aircraft_list = []
        for aircraft in restricted_aircraft:
            severity, time_inside = detector.analyze_severity(aircraft)
            zone_name = next((z['name'] for z in zones if z['id'] == aircraft.zone_id), "Zone")
            aircraft_list.append({
                "id": aircraft.track_id,
                "status": "STATIC" if aircraft.is_static() else "MOVING",
                "time_inside": round(time_inside, 1),
                "severity": severity,
                "zone": zone_name
            })

        alerts = []
        if zone_status == "VIOLATED":
            for aircraft in restricted_aircraft:
                severity, time_inside = detector.analyze_severity(aircraft)
                zone_name = next((z['name'] for z in zones if z['id'] == aircraft.zone_id), "Zone")
                if severity == "CRITICAL":
                    alerts.append({
                        "type": "CRITICAL",
                        "message": f"Aircraft {aircraft.track_id} exceeded {CRITICAL_TIME}s in {zone_name}",
                        "aircraft_id": aircraft.track_id,
                        "severity_color": "red"
                    })
                elif severity == "WARNING":
                    alerts.append({
                        "type": "WARNING",
                        "message": f"Aircraft {aircraft.track_id} staying {round(time_inside, 1)}s in {zone_name}",
                        "aircraft_id": aircraft.track_id,
                        "severity_color": "orange"
                    })

        if aircraft_count >= 2:
            alerts.append({
                "type": "CRITICAL",
                "message": f"Multiple aircraft ({aircraft_count}) in restricted zone",
                "aircraft_id": 0,
                "severity_color": "red"
            })

        past_logs = detector.past_logs[-10:] if detector.past_logs else []
        current_logs = detector.event_logs[-15:] if detector.event_logs else []
        all_logs = past_logs + current_logs

        data = {
            "restricted_zone": {
                "status": zone_status,
                "aircraft_count": aircraft_count,
                "aircraft": aircraft_list,
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
    global current_frame, init_complete
    wait_count = 0
    while is_running:
        with frame_lock:
            frame = current_frame
        
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            wait_count = 0
        else:
            wait_count += 1
            if wait_count > 100:
                print("Warning: No frame available for 3 seconds")
                wait_count = 0
        time.sleep(0.03)


def initialize_system():
    global model, tracker, cap, zones, detector, is_running, processing_thread, init_complete

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return False

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        return False

    if not os.path.exists(ZONES_CONFIG_PATH):
        print(f"Error: Zones config not found at {ZONES_CONFIG_PATH}")
        return False

    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print("Model loaded successfully")

    print("Initializing DeepSort tracker...")
    tracker = DeepSort(max_age=30, n_init=3)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {VIDEO_PATH}")
        return False
    
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("Error: Cannot read video frame")
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"Video opened: {test_frame.shape[1]}x{test_frame.shape[0]}")

    with open(ZONES_CONFIG_PATH, 'r') as f:
        zones = json.load(f)
    print(f"Loaded {len(zones)} zone(s)")

    detector = RestrictedZoneMonitor(zones)
    is_running = True
    processing_thread = threading.Thread(target=process_video)
    processing_thread.daemon = True
    processing_thread.start()
    
    time.sleep(0.5)
    
    _, buffer = cv2.imencode('.jpg', test_frame)
    with frame_lock:
        current_frame = buffer.tobytes()
    
    init_complete = True
    print("System initialized and running")
    return True


@app.route('/')
def index():
    return render_template('restricted_zone.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/data')
def get_data():
    with frame_lock:
        if not current_data:
            return jsonify({
                "restricted_zone": {
                    "status": "CLEAR",
                    "aircraft_count": 0,
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
    global is_running, init_complete
    is_running = False
    init_complete = False
    if cap:
        cap.release()
    return jsonify({"status": "stopped"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
