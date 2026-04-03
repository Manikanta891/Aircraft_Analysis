from flask import Flask, Response, jsonify, render_template, request
import cv2
import numpy as np
import json
import os
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import time
import threading

script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(script_dir).resolve().parent
MODEL_PATH = ROOT_DIR / "Models" / "aircraft_detector_v8.pt"
VIDEO_PATH = ROOT_DIR / "Simulation_Videos" / "runway_single_aircraft.mp4"
RUNWAY_CONFIG_PATH = ROOT_DIR / "Airport_Simulator" / "runways.json"

app = Flask(__name__, template_folder='templates', static_folder='static')

TRAJECTORY_HISTORY_SIZE = 5
SPEED_THRESHOLD = 3.0
ANGLE_TOWARD_THRESHOLD = 30.0
ANGLE_AWAY_THRESHOLD = 60.0
MIN_CONSECUTIVE_FRAMES = 4
MIN_TRAJECTORY_FRAMES = 5

model = None
tracker = None
cap = None
runways = []
fps = 30

detector = None
frame_lock = threading.Lock()
current_frame = None
current_data = {}
is_running = False
processing_thread = None
video_loop_count = 0


def load_runways(filepath: str) -> List[Dict]:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            runways = json.load(f)
            return runways
    return []


def is_point_in_runway(px: float, py: float, runway: dict) -> bool:
    x1 = min(runway['x1'], runway['x2'])
    x2 = max(runway['x1'], runway['x2'])
    y1 = min(runway['y1'], runway['y2'])
    y2 = max(runway['y1'], runway['y2'])
    return x1 <= px <= x2 and y1 <= py <= y2


def is_aircraft_on_any_runway(cx: float, cy: float, runways_list: List[Dict]) -> Tuple[bool, Optional[Dict]]:
    for runway in runways_list:
        if is_point_in_runway(cx, cy, runway):
            return True, runway
    return False, None


@dataclass
class AircraftTrack:
    track_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_SIZE))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_SIZE))
    on_runway: bool = False
    runway_id: Optional[int] = None
    entry_time: Optional[float] = None

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

    def get_direction(self) -> Optional[str]:
        vector = self.get_trajectory_vector()
        if vector is None or vector[2] < SPEED_THRESHOLD:
            return None
        mvx, mvy, _ = vector
        angle = np.arctan2(mvy, mvx) * 180 / np.pi
        if -45 <= angle < 45:
            return "E"
        elif 45 <= angle < 135:
            return "S"
        elif -135 <= angle < -45:
            return "N"
        else:
            return "W"


class RunwayCollisionDetector:
    def __init__(self, runways_list: List[Dict]):
        self.tracks: Dict[int, AircraftTrack] = {}
        self.runways = runways_list
        self.consecutive_frames: Dict[Tuple[int, int], int] = {}
        self.angle_history: Dict[Tuple[int, int], deque] = {}
        self.event_logs: List[Dict] = []
        self.start_time = time.time()
        self.past_events: List[Dict] = []
        self.runway_history: Dict[int, List[Dict]] = {i: [] for i in range(len(runways_list))}

    def reset(self):
        self.tracks.clear()
        self.consecutive_frames.clear()
        self.angle_history.clear()
        self.past_events.extend(self.event_logs)
        self.event_logs.clear()
        self.runway_history = {i: [] for i in range(len(self.runways))}

    def update_track(self, track_id: int, cx: float, cy: float, frame_num: int):
        was_on_runway = False
        if track_id in self.tracks:
            was_on_runway = self.tracks[track_id].on_runway

        if track_id not in self.tracks:
            self.tracks[track_id] = AircraftTrack(track_id=track_id)
            self.tracks[track_id].entry_time = time.time()

        on_runway, runway = is_aircraft_on_any_runway(cx, cy, self.runways)
        self.tracks[track_id].on_runway = on_runway
        self.tracks[track_id].runway_id = self.runways.index(runway) + 1 if runway else None
        self.tracks[track_id].positions.append((cx, cy))
        self.tracks[track_id].timestamps.append(frame_num)

        if on_runway and not was_on_runway:
            self.add_log(f"Aircraft {track_id} entered runway {self.tracks[track_id].runway_id}", "INFO")
        elif not on_runway and was_on_runway:
            self.add_log(f"Aircraft {track_id} exited runway", "INFO")

    def add_log(self, message: str, severity: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.event_logs.append({
            "timestamp": timestamp,
            "message": message,
            "severity": severity
        })
        if len(self.event_logs) > 50:
            self.event_logs = self.event_logs[-50:]

    def get_runway_aircraft(self) -> List[AircraftTrack]:
        return [t for t in self.tracks.values() if t.on_runway and len(t.positions) >= MIN_TRAJECTORY_FRAMES]

    def compute_pair_angle(self, moving_id: int, target_id: int) -> Tuple[Optional[float], bool]:
        moving = self.tracks.get(moving_id)
        target = self.tracks.get(target_id)

        if moving is None or target is None:
            return None, False

        if not moving.on_runway:
            return None, False

        target_pos = target.get_current_position()
        if target_pos is None:
            return None, False

        trajectory = moving.get_trajectory_vector()
        if trajectory is None:
            return None, False

        mvx, mvy, speed = trajectory
        if speed < SPEED_THRESHOLD:
            return None, False

        current = moving.get_current_position()
        if current is None:
            return None, False

        dx = target_pos[0] - current[0]
        dy = target_pos[1] - current[1]
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 1.0:
            return None, False

        dot = (mvx * dx + mvy * dy) / (speed * dist)
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = np.arccos(dot)
        angle_deg = np.degrees(angle_rad)

        pair_key = (moving_id, target_id)
        if pair_key not in self.angle_history:
            self.angle_history[pair_key] = deque(maxlen=TRAJECTORY_HISTORY_SIZE)

        self.angle_history[pair_key].append(float(angle_deg))
        history = list(self.angle_history[pair_key])
        avg_angle = float(np.mean(history))

        return avg_angle, True

    def analyze_collision_risk(self, id_a: int, id_b: int) -> Tuple[str, float, str]:
        track_a = self.tracks.get(id_a)
        track_b = self.tracks.get(id_b)

        if track_a is None or track_b is None:
            return "SAFE", 0.0, "Track not found"

        if not track_a.on_runway or not track_b.on_runway:
            return "SAFE", 0.0, "Aircraft not on runway"

        if track_a.runway_id != track_b.runway_id:
            return "SAFE", 0.0, f"Different runways"

        pair_key = (id_a, id_b)
        reverse_key = (id_b, id_a)

        angle_ab_val, valid_ab = self.compute_pair_angle(id_a, id_b)
        angle_ba_val, valid_ba = self.compute_pair_angle(id_b, id_a)

        if not valid_ab and not valid_ba:
            return "SAFE", 0.0, "Insufficient trajectory data"

        risk_a_toward_b = False
        risk_b_toward_a = False
        angle_ab = angle_ab_val if angle_ab_val is not None else 0.0
        angle_ba = angle_ba_val if angle_ba_val is not None else 0.0

        if valid_ab and angle_ab_val is not None:
            if angle_ab_val < ANGLE_TOWARD_THRESHOLD:
                risk_b_toward_a = True
            elif angle_ab_val > ANGLE_AWAY_THRESHOLD:
                risk_b_toward_a = False

        if valid_ba and angle_ba_val is not None:
            if angle_ba_val < ANGLE_TOWARD_THRESHOLD:
                risk_a_toward_b = True
            elif angle_ba_val > ANGLE_AWAY_THRESHOLD:
                risk_a_toward_b = False

        if risk_a_toward_b:
            if pair_key not in self.consecutive_frames:
                self.consecutive_frames[pair_key] = 0
            self.consecutive_frames[pair_key] += 1

            if reverse_key in self.consecutive_frames:
                self.consecutive_frames[reverse_key] = max(0, self.consecutive_frames[reverse_key] - 1)

            count = self.consecutive_frames[pair_key]
            if count >= MIN_CONSECUTIVE_FRAMES:
                self.add_log(f"Collision risk between ID {id_a} and ID {id_b}", "CRITICAL")
                return "HIGH_RISK", angle_ab, f"A->B: {angle_ab:.1f} deg"
            return "BUILDING", angle_ab, f"A->B: {angle_ab:.1f} deg ({count}/{MIN_CONSECUTIVE_FRAMES})"

        elif risk_b_toward_a:
            if reverse_key not in self.consecutive_frames:
                self.consecutive_frames[reverse_key] = 0
            self.consecutive_frames[reverse_key] += 1

            if pair_key in self.consecutive_frames:
                self.consecutive_frames[pair_key] = max(0, self.consecutive_frames[pair_key] - 1)

            count = self.consecutive_frames[reverse_key]
            if count >= MIN_CONSECUTIVE_FRAMES:
                self.add_log(f"Collision risk between ID {id_a} and ID {id_b}", "CRITICAL")
                return "HIGH_RISK", angle_ba, f"B->A: {angle_ba:.1f} deg"
            return "BUILDING", angle_ba, f"B->A: {angle_ba:.1f} deg ({count}/{MIN_CONSECUTIVE_FRAMES})"

        else:
            if pair_key in self.consecutive_frames:
                self.consecutive_frames[pair_key] = max(0, self.consecutive_frames[pair_key] - 1)
            if reverse_key in self.consecutive_frames:
                self.consecutive_frames[reverse_key] = max(0, self.consecutive_frames[reverse_key] - 1)

            angle = angle_ab if valid_ab else angle_ba
            return "SAFE", angle, f"Angle: {angle:.1f} deg"

    def analyze_runway_pairs(self) -> List[Tuple[int, int, str, float, str]]:
        collision_risks = []
        runway_aircraft = self.get_runway_aircraft()
        runway_ids = [a.track_id for a in runway_aircraft]

        for i, id_a in enumerate(runway_ids):
            for id_b in runway_ids[i + 1:]:
                risk, angle, reason = self.analyze_collision_risk(id_a, id_b)
                collision_risks.append((id_a, id_b, risk, angle, reason))

        return collision_risks

    def cleanup_tracks(self, active_ids: set):
        ids_to_remove = [tid for tid in self.tracks.keys() if tid not in active_ids]
        for tid in ids_to_remove:
            if tid in self.tracks:
                del self.tracks[tid]

        keys_to_remove = [k for k in self.consecutive_frames
                          if k[0] not in active_ids or k[1] not in active_ids]
        for k in keys_to_remove:
            if k in self.consecutive_frames:
                del self.consecutive_frames[k]


def draw_runways_on_frame(frame: np.ndarray, runways_list: List[Dict], highlight_runways: Optional[set] = None):
    for i, runway in enumerate(runways_list):
        x1, y1 = int(runway['x1']), int(runway['y1'])
        x2, y2 = int(runway['x2']), int(runway['y2'])

        if highlight_runways and i in highlight_runways:
            color = (0, 0, 255)
            thickness = 3
        else:
            color = (0, 165, 255)
            thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"RWY {i + 1}"
        cv2.putText(frame, label, (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


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
            
            print(f"Video looped - Resetting detector and tracker (loop #{video_loop_count})")
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
            is_on_runway = track_data.on_runway if track_data else False

            confirmed_tracks.append({
                'id': track_id,
                'cx': cx,
                'cy': cy,
                'l': l, 't': t, 'r': r, 'b': b,
                'on_runway': is_on_runway,
                'runway_id': track_data.runway_id if track_data else None
            })

        detector.cleanup_tracks(active_ids)

        draw_runways_on_frame(frame, runways)

        collision_analysis = detector.analyze_runway_pairs()

        risky_ids = set()
        building_ids = set()
        active_runways = set()

        for id_a, id_b, risk, angle, reason in collision_analysis:
            if risk == "HIGH_RISK":
                risky_ids.add(id_a)
                risky_ids.add(id_b)
                track_a = detector.tracks.get(id_a)
                track_b = detector.tracks.get(id_b)
                if track_a and track_a.runway_id:
                    active_runways.add(track_a.runway_id - 1)
                if track_b and track_b.runway_id:
                    active_runways.add(track_b.runway_id - 1)

        runway_aircraft_count = len([t for t in confirmed_tracks if t['on_runway']])

        for track in confirmed_tracks:
            tid = track['id']
            x1, y1, x2, y2 = int(track['l']), int(track['t']), int(track['r']), int(track['b'])

            track_data = detector.tracks.get(tid)

            if tid in risky_ids:
                color = (0, 0, 255)
                thickness = 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, f"ID:{tid} COLLISION RISK!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            elif tid in building_ids:
                color = (0, 165, 255)
                thickness = 3
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, f"ID:{tid} MONITORING", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            elif track['on_runway']:
                color = (0, 255, 255)
                thickness = 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                status = "STATIC" if (track_data and track_data.is_static()) else "ON RWY"
                cv2.putText(frame, f"ID:{tid} | {status}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            else:
                color = (100, 100, 100)
                thickness = 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, f"ID:{tid} | OFF RWY", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        draw_runways_on_frame(frame, runways, active_runways)

        runway_aircraft = detector.get_runway_aircraft()
        aircraft_list = []
        for track in runway_aircraft:
            aircraft_list.append({
                "id": track.track_id,
                "status": "STATIC" if track.is_static() else "MOVING",
                "direction": track.get_direction() or "N/A",
                "runway_id": f"RWY-{track.runway_id}" if track.runway_id else "N/A",
                "speed": round(track.get_average_speed(), 2)
            })

        collision_risks_list = []
        for id_a, id_b, risk, angle, reason in collision_analysis:
            if risk == "HIGH_RISK" or risk == "BUILDING":
                collision_risks_list.append({
                    "id_a": id_a,
                    "id_b": id_b,
                    "risk": risk,
                    "angle": round(angle, 1),
                    "reason": reason
                })

        runway_count = len(runways)
        runway_statuses = {}
        for i, runway in enumerate(runways):
            runway_num = i + 1
            aircraft_on_this = [a for a in runway_aircraft if a.runway_id == runway_num]
            count = len(aircraft_on_this)
            
            runway_aircraft_ids = [a.track_id for a in aircraft_on_this]
            
            runway_collision_risks = [
                (id_a, id_b, risk, angle, reason) 
                for id_a, id_b, risk, angle, reason in collision_analysis 
                if risk in ["HIGH_RISK", "BUILDING"] and 
                   (id_a in runway_aircraft_ids and id_b in runway_aircraft_ids)
            ]
            
            high_risk = any(risk == "HIGH_RISK" for _, _, risk, _, _ in runway_collision_risks)
            building_risk = any(risk == "BUILDING" for _, _, risk, _, _ in runway_collision_risks)

            if high_risk:
                status = "CRITICAL"
            elif count >= 2 or building_risk:
                status = "UNSAFE"
            elif count == 1:
                status = "ACTIVE"
            else:
                status = "CLEAR"

            utilization = "LOW" if count == 0 else ("MEDIUM" if count == 1 else "HIGH")

            occupancy_time = 0
            if aircraft_on_this:
                entry_times = [a.entry_time for a in aircraft_on_this if a.entry_time]
                if entry_times:
                    occupancy_time = round(time.time() - min(entry_times), 1)

            runway_statuses[f"RWY-{runway_num}"] = {
                "status": status,
                "aircraft_count": count,
                "utilization": utilization,
                "occupancy_time": f"{occupancy_time}s" if occupancy_time > 0 else "0s"
            }

        runway_metrics = {
            "total_aircraft": len(confirmed_tracks),
            "on_runway": runway_aircraft_count,
            "off_runway": len(confirmed_tracks) - runway_aircraft_count,
            "moving": len([a for a in runway_aircraft if not a.is_static()]),
            "static": len([a for a in runway_aircraft if a.is_static()]),
            "avg_speed": round(np.mean([a.get_average_speed() for a in runway_aircraft]) if runway_aircraft else 0, 2),
            "usage_level": "HIGH" if runway_aircraft_count > 3 else ("MEDIUM" if runway_aircraft_count > 1 else "LOW")
        }

        alerts = []
        runway_specific_alerts = {f"RWY-{i+1}": [] for i in range(len(runways))}
        
        for id_a, id_b, risk, angle, reason in collision_analysis:
            track_a = detector.tracks.get(id_a)
            track_b = detector.tracks.get(id_b)
            runway_id = None
            if track_a and track_a.runway_id:
                runway_id = f"RWY-{track_a.runway_id}"
            
            if risk == "HIGH_RISK":
                alerts.append({
                    "type": "CRITICAL",
                    "message": f"Collision risk between ID {id_a} and ID {id_b}",
                    "aircraft_ids": [id_a, id_b],
                    "severity": "red",
                    "runway": runway_id
                })
                if runway_id:
                    runway_specific_alerts[runway_id].append({
                        "type": "CRITICAL",
                        "message": f"Collision between ID {id_a} & ID {id_b}",
                        "aircraft_ids": [id_a, id_b],
                        "severity": "red"
                    })
            elif risk == "BUILDING":
                alerts.append({
                    "type": "WARNING",
                    "message": f"Multiple aircraft on runway - Monitoring {id_a} and {id_b}",
                    "aircraft_ids": [id_a, id_b],
                    "severity": "orange",
                    "runway": runway_id
                })

        for runway_id, aircraft_on_rwy in [(f"RWY-{i+1}", [a for a in runway_aircraft if a.runway_id == i+1]) for i in range(len(runways))]:
            if len(aircraft_on_rwy) >= 2 and not any(a['runway'] == runway_id for a in alerts):
                alerts.append({
                    "type": "WARNING",
                    "message": f"Multiple aircraft ({len(aircraft_on_rwy)}) on {runway_id}",
                    "aircraft_ids": [a.track_id for a in aircraft_on_rwy],
                    "severity": "orange",
                    "runway": runway_id
                })

        past_logs = detector.past_events[-20:] if detector.past_events else []
        current_logs = detector.event_logs[-20:] if detector.event_logs else []
        all_logs = past_logs + current_logs
        
        data = {
            "runway_status": runway_statuses,
            "aircraft": aircraft_list,
            "collision_risks": collision_risks_list,
            "alerts": alerts,
            "logs": all_logs[-30:] if len(all_logs) > 30 else all_logs,
            "metrics": runway_metrics,
            "frame_count": frame_count,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "video_loop": video_loop_count
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
    global model, tracker, cap, runways, detector, is_running, processing_thread

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return False

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        return False

    if not os.path.exists(RUNWAY_CONFIG_PATH):
        print(f"Error: Runway config not found at {RUNWAY_CONFIG_PATH}")
        return False

    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    print("Initializing DeepSort tracker...")
    tracker = DeepSort(max_age=30, n_init=3)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {VIDEO_PATH}")
        return False

    runways = load_runways(str(RUNWAY_CONFIG_PATH))
    print(f"Loaded {len(runways)} runway(s)")

    detector = RunwayCollisionDetector(runways)
    is_running = True
    processing_thread = threading.Thread(target=process_video)
    processing_thread.daemon = True
    processing_thread.start()
    
    print("System initialized and running")

    return True


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/data')
def get_data():
    with frame_lock:
        if not current_data:
            return jsonify({
                "runway_status": {},
                "aircraft": [],
                "collision_risks": [],
                "alerts": [],
                "logs": [],
                "metrics": {"total_aircraft": 0, "on_runway": 0, "off_runway": 0, "moving": 0, "static": 0, "usage_level": "LOW"},
                "frame_count": 0,
                "timestamp": datetime.now().strftime("%H:%M:%S")
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
    app.run(debug=True, host='0.0.0.0', port=5000)
