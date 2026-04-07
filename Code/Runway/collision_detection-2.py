# This code is better then the 1st code

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

import time

script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(script_dir).resolve().parent
MODEL_PATH = ROOT_DIR / "Models" / "aircraft_detector_v8.pt"
VIDEO_PATH = ROOT_DIR / "Simulation_Videos" / "runway_away.mp4"
RUNWAY_CONFIG_PATH = os.path.join(script_dir, "runways.json")

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    exit()

if not os.path.exists(VIDEO_PATH):
    print(f"Error: Video not found at {VIDEO_PATH}")
    exit()

print(f"Loading model from: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

print("Initializing DeepSort tracker...")
tracker = DeepSort(max_age=30, n_init=3)

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    print(f"Error: Cannot open video: {VIDEO_PATH}")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

TRAJECTORY_HISTORY_SIZE = 5
SPEED_THRESHOLD = 3.0
ANGLE_TOWARD_THRESHOLD = 30.0
ANGLE_AWAY_THRESHOLD = 60.0
MIN_CONSECUTIVE_FRAMES = 4
MIN_TRAJECTORY_FRAMES = 5

output_path = os.path.join(script_dir, "runway_collision_output.avi")
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (video_w, video_h))


def load_runways(filepath: str) -> List[Dict]:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            runways = json.load(f)
            print(f"Loaded {len(runways)} runway(s) from: {filepath}")
            return runways
    print(f"Warning: No runway config found at {filepath}")
    print("Runway collision detection DISABLED - all aircraft will be ignored")
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


runways = load_runways(RUNWAY_CONFIG_PATH)

print("\n" + "=" * 60)
print("RUNWAY-BASED COLLISION RISK DETECTION SYSTEM")
print("=" * 60)
print(f"Video: {VIDEO_PATH}")
print(f"Resolution: {video_w}x{video_h}")
print(f"Number of runways: {len(runways)}")
for i, r in enumerate(runways):
    print(f"  Runway {i+1}: ({r['x1']},{r['y1']})-({r['x2']},{r['y2']})")
print(f"Trajectory History: {TRAJECTORY_HISTORY_SIZE} frames")
print(f"Speed Threshold: {SPEED_THRESHOLD} pixels/frame")
print(f"Angle Toward: < {ANGLE_TOWARD_THRESHOLD} degrees")
print(f"Angle Away: > {ANGLE_AWAY_THRESHOLD} degrees")
print(f"Min Consecutive Frames: {MIN_CONSECUTIVE_FRAMES}")
print("=" * 60)


@dataclass
class AircraftTrack:
    track_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_SIZE))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_SIZE))
    on_runway: bool = False
    runway_id: Optional[int] = None

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

    def get_direction_angle(self, target_pos: Tuple[float, float]) -> Optional[float]:
        trajectory = self.get_trajectory_vector()
        if trajectory is None:
            return None

        mvx, mvy, speed = trajectory
        if speed < SPEED_THRESHOLD:
            return None

        current = self.get_current_position()
        if current is None:
            return None

        dx = target_pos[0] - current[0]
        dy = target_pos[1] - current[1]
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 1.0:
            return None

        dot = (mvx * dx + mvy * dy) / (speed * dist)
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = np.arccos(dot)
        angle_deg = np.degrees(angle_rad)

        return angle_deg


class RunwayCollisionDetector:
    def __init__(self, runways_list: List[Dict]):
        self.tracks: Dict[int, AircraftTrack] = {}
        self.runways = runways_list
        self.consecutive_frames: Dict[Tuple[int, int], int] = {}
        self.angle_history: Dict[Tuple[int, int], deque] = {}

    def update_track(self, track_id: int, cx: float, cy: float, frame_num: int):
        if track_id not in self.tracks:
            self.tracks[track_id] = AircraftTrack(track_id=track_id)

        on_runway, runway = is_aircraft_on_any_runway(cx, cy, self.runways)
        self.tracks[track_id].on_runway = on_runway
        self.tracks[track_id].runway_id = self.runways.index(runway) + 1 if runway else None

        self.tracks[track_id].positions.append((cx, cy))
        self.tracks[track_id].timestamps.append(frame_num)

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

        angle = moving.get_direction_angle(target_pos)

        if angle is None:
            return None, False

        pair_key = (moving_id, target_id)
        if pair_key not in self.angle_history:
            self.angle_history[pair_key] = deque(maxlen=TRAJECTORY_HISTORY_SIZE)

        self.angle_history[pair_key].append(float(angle))

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
            return "SAFE", 0.0, f"Different runways (A:RWY{track_a.runway_id}, B:RWY{track_b.runway_id})"

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
                return "HIGH_RISK", angle_ab, f"A->B: {angle_ab:.1f} deg for {count} frames"
            return "BUILDING", angle_ab, f"A->B: {angle_ab:.1f} deg ({count}/{MIN_CONSECUTIVE_FRAMES})"

        elif risk_b_toward_a:
            if reverse_key not in self.consecutive_frames:
                self.consecutive_frames[reverse_key] = 0
            self.consecutive_frames[reverse_key] += 1

            if pair_key in self.consecutive_frames:
                self.consecutive_frames[pair_key] = max(0, self.consecutive_frames[pair_key] - 1)

            count = self.consecutive_frames[reverse_key]
            if count >= MIN_CONSECUTIVE_FRAMES:
                return "HIGH_RISK", angle_ba, f"B->A: {angle_ba:.1f} deg for {count} frames"
            return "BUILDING", angle_ba, f"B->A: {angle_ba:.1f} deg ({count}/{MIN_CONSECUTIVE_FRAMES})"

        else:
            if pair_key in self.consecutive_frames:
                self.consecutive_frames[pair_key] = max(0, self.consecutive_frames[pair_key] - 1)
            if reverse_key in self.consecutive_frames:
                self.consecutive_frames[reverse_key] = max(0, self.consecutive_frames[reverse_key] - 1)

            angle = angle_ab if valid_ab else angle_ba
            return "SAFE", angle, f"Angle: {angle:.1f} deg (not aligned)"

    def analyze_runway_pairs(self) -> List[Tuple[int, int, str, float, str]]:
        collision_risks = []
        runway_aircraft = self.get_runway_aircraft()
        runway_ids = [a.track_id for a in runway_aircraft]

        for i, id_a in enumerate(runway_ids):
            for id_b in runway_ids[i + 1:]:
                risk, angle, reason = self.analyze_collision_risk(id_a, id_b)
                collision_risks.append((id_a, id_b, risk, angle, reason))

        return collision_risks

    def get_risky_track_ids(self) -> Set[int]:
        risky = set()
        for id_a, id_b, risk, _, _ in self.analyze_runway_pairs():
            if risk == "HIGH_RISK":
                risky.add(id_a)
                risky.add(id_b)
        return risky

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


detector = RunwayCollisionDetector(runways)

frame_count = 0
collision_warnings = 0


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

start_time = time.time()
frame_times = []
# define before loop
prev_results = None
while True:
    frame_start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 🔥 ADD THIS LINE HERE
    current_detections = []

    # -----------------------------
    # YOLO (frame skipping)
    # -----------------------------
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
    risk_details = {}
    active_runways = set()

    for id_a, id_b, risk, angle, reason in collision_analysis:
        if risk == "HIGH_RISK":
            risky_ids.add(id_a)
            risky_ids.add(id_b)
            risk_details[id_a] = (risk, reason, angle)
            risk_details[id_b] = (risk, reason, angle)

            track_a = detector.tracks.get(id_a)
            track_b = detector.tracks.get(id_b)
            if track_a and track_a.runway_id:
                active_runways.add(track_a.runway_id - 1)
            if track_b and track_b.runway_id:
                active_runways.add(track_b.runway_id - 1)

            collision_warnings += 1
            print(f"[!] RUNWAY COLLISION RISK between ID {id_a} and ID {id_b} | {reason}")

        elif risk == "BUILDING":
            building_ids.add(id_a)
            building_ids.add(id_b)
            if id_a not in risk_details:
                risk_details[id_a] = (risk, reason, angle)
            if id_b not in risk_details:
                risk_details[id_b] = (risk, reason, angle)

    runway_aircraft_count = len([t for t in confirmed_tracks if t['on_runway']])
    non_runway_aircraft_count = len(confirmed_tracks) - runway_aircraft_count

    for track in confirmed_tracks:
        tid = track['id']
        x1, y1, x2, y2 = int(track['l']), int(track['t']), int(track['r']), int(track['b'])
        cx, cy = int(track['cx']), int(track['cy'])

        track_data = detector.tracks.get(tid)
        trajectory = track_data.get_trajectory_vector() if track_data else None

        if tid in risky_ids:
            color = (0, 0, 255)
            thickness = 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            risk_info = risk_details.get(tid, ("RISK", "Unknown", 0))
            cv2.putText(frame, f"ID:{tid} RUNWAY COLLISION!", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"{risk_info[1][:35]}", (x1, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if track_data:
                cv2.putText(frame, f"RWY:{track_data.runway_id}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        elif tid in building_ids:
            color = (0, 165, 255)
            thickness = 3
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            risk_info = risk_details.get(tid, ("BUILDING", "Monitoring", 0))
            cv2.putText(frame, f"ID:{tid} MONITORING", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"{risk_info[1][:30]}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if track_data:
                cv2.putText(frame, f"RWY:{track_data.runway_id}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        elif track['on_runway']:
            color = (0, 255, 255)
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            status = "STATIC" if (track_data and track_data.is_static()) else "ON RWY"
            cv2.putText(frame, f"ID:{tid} | {status}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if track_data:
                cv2.putText(frame, f"RWY:{track_data.runway_id}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if trajectory and track_data and not track_data.is_static():
                draw_trajectory_arrow(frame, (cx, cy), trajectory, (255, 255, 0), 0.6)

        else:
            color = (100, 100, 100)
            thickness = 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"ID:{tid} | OFF RWY", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for id_a, id_b, risk, angle, reason in collision_analysis:
        if risk == "HIGH_RISK":
            track_a = next((t for t in confirmed_tracks if t['id'] == id_a), None)
            track_b = next((t for t in confirmed_tracks if t['id'] == id_b), None)
            if track_a and track_b:
                cv2.line(frame,
                        (int(track_a['cx']), int(track_a['cy'])),
                        (int(track_b['cx']), int(track_b['cy'])),
                        (0, 0, 255), 3)

    draw_runways_on_frame(frame, runways, active_runways)

    panel_h = 170
    # cv2.rectangle(frame, (0, 0), (500, panel_h), (0, 0, 0), -1)
    # cv2.putText(frame, "RUNWAY-BASED COLLISION DETECTION", (10, 25),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(frame, f"Total Tracks: {len(confirmed_tracks)} | Frame: {frame_count}", (10, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    # cv2.putText(frame, f"On Runway: {runway_aircraft_count} | Off Runway: {non_runway_aircraft_count}", (10, 75),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if risky_ids:
        cv2.putText(frame, f"ALERT: RUNWAY COLLISION RISK DETECTED!", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif building_ids:
        cv2.putText(frame, f"Caution: {len(building_ids)} Aircraft(s) Being Monitored", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    else:
        cv2.putText(frame, f"Status: RUNWAYS CLEAR", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # cv2.putText(frame, f"Warnings: {collision_warnings} | Runways: {len(runways)}", (10, 125),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    # cv2.putText(frame, f"Consecutive: {MIN_CONSECUTIVE_FRAMES}f | Angle: {ANGLE_TOWARD_THRESHOLD} deg", (10, 150),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    legend_y = panel_h + 20
    # cv2.rectangle(frame, (video_w - 220, legend_y - 20), (video_w - 10, legend_y + 100), (20, 20, 20), -1)
    # cv2.putText(frame, "LEGEND", (video_w - 210, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    # cv2.rectangle(frame, (video_w - 210, legend_y + 10), (video_w - 190, legend_y + 25), (0, 0, 255), -1)
    # cv2.putText(frame, "Collision Risk", (video_w - 185, legend_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    # cv2.rectangle(frame, (video_w - 210, legend_y + 30), (video_w - 190, legend_y + 45), (0, 165, 255), -1)
    # cv2.putText(frame, "Monitoring", (video_w - 185, legend_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)
    # cv2.rectangle(frame, (video_w - 210, legend_y + 50), (video_w - 190, legend_y + 65), (0, 255, 255), -1)
    # cv2.putText(frame, "On Runway", (video_w - 185, legend_y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    # cv2.rectangle(frame, (video_w - 210, legend_y + 70), (video_w - 190, legend_y + 85), (100, 100, 100), -1)
    # cv2.putText(frame, "Off Runway", (video_w - 185, legend_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    frame_end = time.time()     # ⏱️ END FRAME TIMER
    frame_time = frame_end - frame_start
    frame_times.append(frame_time)

    # Show FPS on screen (real-time)
    fps_live = 1.0 / frame_time if frame_time > 0 else 0
    cv2.putText(frame,
                f"FPS: {fps_live:.2f}",
                (10, video_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2)

    cv2.imshow("Runway-Based Collision Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

end_time = time.time()

total_time = end_time - start_time
total_frames = frame_count

avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
fps = total_frames / total_time if total_time > 0 else 0

print("\n" + "=" * 60)
print("PERFORMANCE ANALYSIS")
print("=" * 60)
print(f"Total Frames: {total_frames}")
print(f"Total Time: {total_time:.2f} sec")
print(f"Average Frame Time: {avg_frame_time:.4f} sec")
print(f"Processing FPS: {fps:.2f}")
print("=" * 60)
