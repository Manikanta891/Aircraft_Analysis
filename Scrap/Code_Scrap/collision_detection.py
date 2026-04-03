import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(script_dir).resolve().parent
MODEL_PATH = ROOT_DIR / "Models" / "aircraft_detector_v8.pt"
VIDEO_PATH = ROOT_DIR / "Simulation_Videos" / "collision_video.mp4"

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

TRAJECTORY_HISTORY_SIZE = 8
SPEED_THRESHOLD = 3.0
ANGLE_TOWARD_THRESHOLD = 30.0
ANGLE_AWAY_THRESHOLD = 60.0
MIN_CONSECUTIVE_FRAMES = 4
MIN_TRAJECTORY_FRAMES = 5

output_path = os.path.join(script_dir, "trajectory_collision_output.avi")
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (video_w, video_h))

print("\n" + "=" * 60)
print("AIRCRAFT TRAJECTORY-BASED COLLISION RISK DETECTION")
print("=" * 60)
print(f"Video: {VIDEO_PATH}")
print(f"Resolution: {video_w}x{video_h}")
print(f"Trajectory History: {TRAJECTORY_HISTORY_SIZE} frames")
print(f"Speed Threshold: {SPEED_THRESHOLD} pixels/frame")
print(f"Angle Toward: < {ANGLE_TOWARD_THRESHOLD} degrees")
print(f"Angle Away: > {ANGLE_AWAY_THRESHOLD} degrees")
print(f"Min Consecutive Frames for Risk: {MIN_CONSECUTIVE_FRAMES}")
print("=" * 60)


@dataclass
class AircraftTrack:
    track_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_SIZE))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_SIZE))

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


class TrajectoryCollisionDetector:
    def __init__(self):
        self.tracks: Dict[int, AircraftTrack] = {}
        self.consecutive_frames: Dict[Tuple[int, int], int] = {}
        self.angle_history: Dict[Tuple[int, int], deque] = {}

    def update_track(self, track_id: int, cx: float, cy: float, frame_num: int):
        if track_id not in self.tracks:
            self.tracks[track_id] = AircraftTrack(track_id=track_id)

        self.tracks[track_id].positions.append((cx, cy))
        self.tracks[track_id].timestamps.append(frame_num)

    def compute_pair_angle(self, moving_id: int, target_id: int) -> Tuple[Optional[float], bool]:
        moving = self.tracks.get(moving_id)
        target = self.tracks.get(target_id)

        if moving is None or target is None:
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

    def analyze_all_pairs(self) -> List[Tuple[int, int, str, float, str]]:
        collision_risks = []
        track_ids = list(self.tracks.keys())

        for i, id_a in enumerate(track_ids):
            for id_b in track_ids[i + 1:]:
                risk, angle, reason = self.analyze_collision_risk(id_a, id_b)
                collision_risks.append((id_a, id_b, risk, angle, reason))

        return collision_risks

    def get_risky_track_ids(self) -> set:
        risky = set()
        for id_a, id_b, risk, _, _ in self.analyze_all_pairs():
            if risk == "HIGH_RISK":
                risky.add(id_a)
                risky.add(id_b)
        return risky

    def get_track_details(self, track_id: int) -> Optional[Dict]:
        track = self.tracks.get(track_id)
        if track is None:
            return None

        trajectory = track.get_trajectory_vector()
        return {
            'id': track_id,
            'position': track.get_current_position(),
            'trajectory': trajectory,
            'speed': track.get_average_speed(),
            'is_static': track.is_static()
        }

    def cleanup_tracks(self, active_ids: set):
        ids_to_remove = [tid for tid in self.tracks.keys() if tid not in active_ids]
        for tid in ids_to_remove:
            if tid in self.tracks:
                del self.tracks[tid]
            if tid in self.angle_history:
                del self.angle_history[tid]

        keys_to_remove = [k for k in self.consecutive_frames if k[0] not in active_ids or k[1] not in active_ids]
        for k in keys_to_remove:
            del self.consecutive_frames[k]


detector = TrajectoryCollisionDetector()

frame_count = 0
collision_warnings = 0


def draw_trajectory_arrow(frame, start: Tuple[int, int], vector: Tuple[float, float, float],
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


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_detections = []

    results = model(frame, conf=0.05, imgsz=1280)[0]

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

        confirmed_tracks.append({
            'id': track_id,
            'cx': cx,
            'cy': cy,
            'l': l, 't': t, 'r': r, 'b': b
        })

    detector.cleanup_tracks(active_ids)

    collision_analysis = detector.analyze_all_pairs()

    risky_ids = set()
    building_ids = set()
    risk_details = {}

    for id_a, id_b, risk, angle, reason in collision_analysis:
        if risk == "HIGH_RISK":
            risky_ids.add(id_a)
            risky_ids.add(id_b)
            risk_details[id_a] = (risk, reason, angle)
            risk_details[id_b] = (risk, reason, angle)
            collision_warnings += 1
            print(f"[!] Stable Directional Collision Risk between ID {id_a} and ID {id_b} | {reason}")

        elif risk == "BUILDING":
            building_ids.add(id_a)
            building_ids.add(id_b)
            if id_a not in risk_details:
                risk_details[id_a] = (risk, reason, angle)
            if id_b not in risk_details:
                risk_details[id_b] = (risk, reason, angle)

    for track in confirmed_tracks:
        tid = track['id']
        x1, y1, x2, y2 = int(track['l']), int(track['t']), int(track['r']), int(track['b'])
        cx, cy = int(track['cx']), int(track['cy'])

        track_data = detector.get_track_details(tid)

        if tid in risky_ids:
            color = (0, 0, 255)
            thickness = 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            risk_info = risk_details.get(tid, ("RISK", "Unknown", 0))
            cv2.putText(frame, f"ID:{tid} HIGH RISK!", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"{risk_info[1][:35]}", (x1, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        elif tid in building_ids:
            color = (0, 165, 255)
            thickness = 3
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            risk_info = risk_details.get(tid, ("BUILDING", "Monitoring", 0))
            cv2.putText(frame, f"ID:{tid} MONITORING", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"{risk_info[1][:30]}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        else:
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            status = "STATIC" if (track_data and track_data['is_static']) else "SAFE"
            cv2.putText(frame, f"ID:{tid} | {status}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if track_data and track_data['trajectory']:
            draw_trajectory_arrow(frame, (cx, cy), track_data['trajectory'], (255, 255, 0), 0.6)

    for id_a, id_b, risk, angle, reason in collision_analysis:
        if risk == "HIGH_RISK":
            track_a = next((t for t in confirmed_tracks if t['id'] == id_a), None)
            track_b = next((t for t in confirmed_tracks if t['id'] == id_b), None)
            if track_a and track_b:
                cv2.line(frame,
                        (int(track_a['cx']), int(track_a['cy'])),
                        (int(track_b['cx']), int(track_b['cy'])),
                        (0, 0, 255), 2)

    panel_h = 150
    cv2.rectangle(frame, (0, 0), (450, panel_h), (0, 0, 0), -1)
    cv2.putText(frame, "TRAJECTORY-BASED COLLISION DETECTION", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f"Tracks: {len(confirmed_tracks)} | Frame: {frame_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if risky_ids:
        cv2.putText(frame, f"ALERT: {len(risky_ids)} HIGH RISK AIRCRAFT(S)!", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif building_ids:
        cv2.putText(frame, f"Caution: {len(building_ids)} AIRCRAFT(S) BEING MONITORED", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    else:
        cv2.putText(frame, "Status: ALL SAFE", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Total Warnings: {collision_warnings}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.putText(frame, f"Trajectory: {TRAJECTORY_HISTORY_SIZE}f | Consecutive: {MIN_CONSECUTIVE_FRAMES}f", (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    legend_y = panel_h + 20
    cv2.rectangle(frame, (video_w - 210, legend_y - 20), (video_w - 10, legend_y + 80), (20, 20, 20), -1)
    cv2.putText(frame, "LEGEND", (video_w - 200, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(frame, (video_w - 200, legend_y + 10), (video_w - 180, legend_y + 25), (0, 0, 255), -1)
    cv2.putText(frame, "High Risk", (video_w - 175, legend_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.rectangle(frame, (video_w - 200, legend_y + 30), (video_w - 180, legend_y + 45), (0, 165, 255), -1)
    cv2.putText(frame, "Monitoring", (video_w - 175, legend_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)
    cv2.rectangle(frame, (video_w - 200, legend_y + 50), (video_w - 180, legend_y + 65), (0, 255, 0), -1)
    cv2.putText(frame, "Safe", (video_w - 175, legend_y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    cv2.imshow("Trajectory-Based Aircraft Collision Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("TRAJECTORY-BASED COLLISION DETECTION COMPLETE")
print("=" * 60)
print(f"Total frames processed: {frame_count}")
print(f"Total collision warnings: {collision_warnings}")
print(f"Output saved: {output_path}")
print("=" * 60)
