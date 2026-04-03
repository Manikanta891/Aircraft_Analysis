import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import csv
import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "Models" / "aircraft_detector_v8.pt"
VIDEO_PATH = ROOT_DIR / "Simulation_Videos" / "restricted_area_simulation.mp4"
RESTRICTED_ZONES_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "restricted_zones.json")

TRAJECTORY_HISTORY_SIZE = 8
MIN_TRAJECTORY_FRAMES = 5
ANGLE_TOWARD_THRESHOLD = 45.0
SPEED_THRESHOLD = 3.0
WARNING_CONSECUTIVE_FRAMES = 3


@dataclass
class AircraftTrack:
    track_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_SIZE))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_SIZE))
    in_restricted_zone: bool = False
    in_warning_zone: bool = False
    zone_id: Optional[int] = None
    consecutive_warnings: int = 0
    trajectory_history: deque = field(default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_SIZE))

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

    def is_moving(self) -> bool:
        return self.get_average_speed() >= SPEED_THRESHOLD

    def get_direction_toward_zone(self, zone_center: Tuple[float, float]) -> Optional[float]:
        trajectory = self.get_trajectory_vector()
        if trajectory is None:
            return None

        mvx, mvy, speed = trajectory
        if speed < SPEED_THRESHOLD:
            return None

        current = self.get_current_position()
        if current is None:
            return None

        dx = zone_center[0] - current[0]
        dy = zone_center[1] - current[1]
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 1.0:
            return None

        dot = (mvx * dx + mvy * dy) / (speed * dist)
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = np.arccos(dot)
        angle_deg = np.degrees(angle_rad)

        return angle_deg


class RestrictedZoneWarningSystem:
    def __init__(self, zones: List[Dict]):
        self.zones = zones
        self.tracks: Dict[int, AircraftTrack] = {}
        self.warning_zone_ids: Set[int] = set()
        self.total_warnings = 0

    def is_point_in_box(self, px: float, py: float, box: dict) -> bool:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)

    def get_zone_center(self, zone: dict) -> Tuple[float, float]:
        r = zone['restricted']
        return ((r['x1'] + r['x2']) / 2, (r['y1'] + r['y2']) / 2)

    def check_zone_status(self, cx: float, cy: float) -> Tuple[bool, bool, Optional[Dict]]:
        for zone in self.zones:
            if self.is_point_in_box(cx, cy, zone['restricted']):
                return True, False, zone
            if self.is_point_in_box(cx, cy, zone['warning']):
                return False, True, zone
        return False, False, None

    def update_track(self, track_id: int, cx: float, cy: float, frame_num: int):
        if track_id not in self.tracks:
            self.tracks[track_id] = AircraftTrack(track_id=track_id)

        in_restricted, in_warning, zone = self.check_zone_status(cx, cy)
        self.tracks[track_id].in_restricted_zone = in_restricted
        self.tracks[track_id].in_warning_zone = in_warning
        self.tracks[track_id].zone_id = zone['id'] if zone else None

        self.tracks[track_id].positions.append((cx, cy))
        self.tracks[track_id].timestamps.append(frame_num)

    def analyze_approach_warning(self, track_id: int) -> Tuple[bool, Optional[float], Optional[Dict]]:
        track = self.tracks.get(track_id)
        if track is None:
            return False, None, None

        if track.in_restricted_zone or track.in_warning_zone:
            return False, None, None

        if len(track.positions) < MIN_TRAJECTORY_FRAMES:
            return False, None, None

        if not track.is_moving():
            return False, None, None

        for zone in self.zones:
            zone_center = self.get_zone_center(zone)
            angle = track.get_direction_toward_zone(zone_center)

            if angle is not None and angle < ANGLE_TOWARD_THRESHOLD:
                track.consecutive_warnings += 1
                track.trajectory_history.append(angle)

                if track.consecutive_warnings >= WARNING_CONSECUTIVE_FRAMES:
                    return True, angle, zone
        return False, None, None

    def analyze_all_tracks(self) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
        approaching = []
        in_warning = []
        in_restricted = []
        self.warning_zone_ids.clear()

        for track_id in list(self.tracks.keys()):
            track = self.tracks.get(track_id)
            if track is None:
                continue

            if track.in_restricted_zone and track.zone_id:
                zone = next((z for z in self.zones if z['id'] == track.zone_id), None)
                if zone:
                    in_restricted.append((track_id, zone))
                    self.warning_zone_ids.add(track.zone_id)
            elif track.in_warning_zone and track.zone_id:
                zone = next((z for z in self.zones if z['id'] == track.zone_id), None)
                if zone:
                    in_warning.append((track_id, zone))
                    self.warning_zone_ids.add(track.zone_id)
            else:
                is_approach, angle, zone = self.analyze_approach_warning(track_id)
                if is_approach and zone:
                    approaching.append((track_id, angle, zone))
                    self.warning_zone_ids.add(zone['id'])
                    print(f"[!] WARNING: Aircraft ID {track_id} approaching '{zone['name']}' | Angle: {angle:.1f}°")

        return approaching, in_warning, in_restricted

    def cleanup_tracks(self, active_ids: set):
        ids_to_remove = [tid for tid in self.tracks.keys() if tid not in active_ids]
        for tid in ids_to_remove:
            if tid in self.tracks:
                del self.tracks[tid]


zones = []
if os.path.exists(RESTRICTED_ZONES_CONFIG):
    with open(RESTRICTED_ZONES_CONFIG, 'r') as f:
        zones = json.load(f)
        print(f"Loaded {len(zones)} zone(s)")
else:
    zones = []
    print("Warning: No restricted zones config found!")

model = YOLO(str(MODEL_PATH))
tracker = DeepSort(max_age=10, n_init=3)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print(f"Error: Cannot open video: {VIDEO_PATH}")
    exit()

video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video loaded: {video_w}x{video_h}")

cv2.namedWindow("Aircraft Restricted Zone Monitoring", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Aircraft Restricted Zone Monitoring", video_w, video_h)

warning_system = RestrictedZoneWarningSystem(zones)

aircraft_log = {}
event_log = []
EXIT_THRESHOLD = 3

frame_count = 0


def draw_zones(frame, zones_list, active_zones=None):
    if active_zones is None:
        active_zones = set()

    for zone in zones_list:
        r = zone['restricted']
        w = zone['warning']
        
        r_x1, r_y1 = int(r['x1']), int(r['y1'])
        r_x2, r_y2 = int(r['x2']), int(r['y2'])
        w_x1, w_y1 = int(w['x1']), int(w['y1'])
        w_x2, w_y2 = int(w['x2']), int(w['y2'])

        is_active = zone['id'] in active_zones

        overlay = frame.copy()
        cv2.rectangle(overlay, (w_x1, w_y1), (w_x2, w_y2), (0, 165, 255), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.rectangle(frame, (w_x1, w_y1), (w_x2, w_y2), (0, 165, 255), 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (r_x1, r_y1), (r_x2, r_y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.rectangle(frame, (r_x1, r_y1), (r_x2, r_y2), (0, 0, 255), 3)

        cx = (r_x1 + r_x2) // 2
        cy = r_y1 - 10
        color = (0, 0, 255) if is_active else (255, 0, 255)
        cv2.putText(frame, zone['name'], (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_trajectory_arrow(frame, start, vector, color, length_scale=0.5):
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
    current_time = time.time()

    results = model(frame, conf=0.15, imgsz=960)[0]

    detections = []

    if results.boxes is not None:
        for r in results.boxes.data:
            x1, y1, x2, y2, conf, cls = r.tolist()
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], conf, 'aircraft'))

    tracks = tracker.update_tracks(detections, frame=frame)

    active_ids = set()
    confirmed_tracks = []

    for track in tracks:
        if not track.is_confirmed() or track.hits < 3:
            continue

        track_id = track.track_id
        active_ids.add(track_id)

        l, t, r, b = track.to_ltrb()
        cx = (l + r) / 2
        cy = (t + b) / 2

        warning_system.update_track(track_id, cx, cy, frame_count)

        track_data = warning_system.tracks.get(track_id)
        in_restricted = track_data.in_restricted_zone if track_data else False
        in_warning = track_data.in_warning_zone if track_data else False

        confirmed_tracks.append({
            'id': track_id,
            'cx': cx,
            'cy': cy,
            'l': l, 't': t, 'r': r, 'b': b,
            'in_restricted': in_restricted,
            'in_warning': in_warning,
            'zone_id': track_data.zone_id if track_data else None,
            'track_data': track_data
        })

    warning_system.cleanup_tracks(active_ids)

    draw_zones(frame, zones, warning_system.warning_zone_ids)

    approaching, in_warning_zone, in_restricted_zone = warning_system.analyze_all_tracks()
    approaching_ids = {a[0] for a in approaching}
    warning_zone_ids = {w[0] for w in in_warning_zone}
    restricted_zone_ids = {r[0] for r in in_restricted_zone}

    for track in confirmed_tracks:
        tid = track['id']
        x1, y1, x2, y2 = int(track['l']), int(track['t']), int(track['r']), int(track['b'])
        cx, cy = int(track['cx']), int(track['cy'])
        track_data = track['track_data']
        trajectory = track_data.get_trajectory_vector() if track_data else None

        if tid in restricted_zone_ids:
            color = (0, 0, 255)
            thickness = 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"ID:{tid} - IN RESTRICTED ZONE!", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            zone = next((z for z in zones if z['id'] == track['zone_id']), None)
            if zone:
                cv2.putText(frame, f"{zone['name']}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            event_key = f"{tid}_restricted"
            if event_key not in aircraft_log:
                aircraft_log[event_key] = {
                    "event": "IN RESTRICTED ZONE",
                    "track_id": tid,
                    "zone": zone['name'] if zone else "Unknown",
                    "start_time": current_time,
                    "last_seen": current_time,
                    "active": True
                }

        elif tid in warning_zone_ids:
            color = (0, 165, 255)
            thickness = 3
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"ID:{tid} - IN WARNING ZONE", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            zone = next((z for z in zones if z['id'] == track['zone_id']), None)
            if zone:
                cv2.putText(frame, f"{zone['name']}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            event_key = f"{tid}_warning"
            if event_key not in aircraft_log:
                aircraft_log[event_key] = {
                    "event": "IN WARNING ZONE",
                    "track_id": tid,
                    "zone": zone['name'] if zone else "Unknown",
                    "start_time": current_time,
                    "last_seen": current_time,
                    "active": True
                }

        elif tid in approaching_ids:
            color = (0, 0, 255)
            thickness = 3
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"ID:{tid} - APPROACHING!", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if trajectory:
                draw_trajectory_arrow(frame, (cx, cy), trajectory, (0, 255, 255), 0.8)
            zone = next((z for z in zones if any(a[0] == tid and a[2]['id'] == z['id'] for a in approaching)), None)
            if zone:
                cv2.putText(frame, f"-> {zone['name']}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            event_key = f"{tid}_approaching"
            if event_key not in aircraft_log:
                aircraft_log[event_key] = {
                    "event": "APPROACHING",
                    "track_id": tid,
                    "zone": zone['name'] if zone else "Unknown",
                    "start_time": current_time,
                    "last_seen": current_time,
                    "active": True
                }

        else:
            color = (255, 0, 0)
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"ID:{tid}", (cx - 15, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if trajectory and track_data and track_data.is_moving():
                draw_trajectory_arrow(frame, (cx, cy), trajectory, (0, 255, 0), 0.5)

        for key in list(aircraft_log.keys()):
            if key.startswith(f"{tid}_"):
                aircraft_log[key]["last_seen"] = current_time

    for key in list(aircraft_log.keys()):
        tid_part = key.split('_')[0]
        if int(tid_part) not in active_ids and aircraft_log[key]["active"]:
            if current_time - aircraft_log[key]["last_seen"] > EXIT_THRESHOLD:
                aircraft_log[key]["end_time"] = current_time
                aircraft_log[key]["duration"] = aircraft_log[key]["end_time"] - aircraft_log[key]["start_time"]
                aircraft_log[key]["active"] = False

                event_log.append({
                    "track_id": aircraft_log[key]["track_id"],
                    "event": aircraft_log[key]["event"],
                    "zone": aircraft_log[key]["zone"],
                    "start_time": time.strftime('%H:%M:%S', time.localtime(aircraft_log[key]["start_time"])),
                    "end_time": time.strftime('%H:%M:%S', time.localtime(aircraft_log[key]["end_time"])),
                    "duration": round(aircraft_log[key]["duration"], 2)
                })

    total_now = len(active_ids)
    restricted_count = len(in_restricted_zone)
    warning_count = len(in_warning_zone)
    approaching_count = len(approaching)

    panel_w = min(450, frame.shape[1] - 20)
    panel_h = 180
    
    panel_overlay = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    cv2.rectangle(panel_overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    frame[0:panel_h, 0:panel_w] = cv2.addWeighted(panel_overlay, 0.6, frame[0:panel_h, 0:panel_w], 0.4, 0)

    cv2.putText(frame, "AIRCRAFT RESTRICTED ZONE MONITORING", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_count} | Total Aircraft: {total_now}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, f"In Restricted Zone: {restricted_count}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(frame, f"In Warning Zone: {warning_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
    cv2.putText(frame, f"Approaching: {approaching_count}", (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    if restricted_count > 0:
        cv2.putText(frame, "! INTRUSION DETECTED !", (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif warning_count > 0:
        cv2.putText(frame, "WARNING: IN RESTRICTED AREA", (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    elif approaching_count > 0:
        cv2.putText(frame, "CAUTION: AIRCRAFT APPROACHING", (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Status: ALL CLEAR", (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    legend_x = max(10, frame.shape[1] - 180)
    legend_w = 170
    legend_h = 130
    legend_overlay = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
    cv2.rectangle(legend_overlay, (0, 0), (legend_w, legend_h), (0, 0, 0), -1)
    frame[10:10+legend_h, legend_x:legend_x+legend_w] = cv2.addWeighted(
        legend_overlay, 0.6, frame[10:10+legend_h, legend_x:legend_x+legend_w], 0.4, 0)

    cv2.putText(frame, "LEGEND", (legend_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(frame, (legend_x, 45), (legend_x + 20, 58), (0, 0, 255), -1)
    cv2.putText(frame, "In Restricted", (legend_x + 25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.rectangle(frame, (legend_x, 68), (legend_x + 20, 81), (0, 165, 255), -1)
    cv2.putText(frame, "In Warning", (legend_x + 25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)
    cv2.rectangle(frame, (legend_x, 91), (legend_x + 20, 104), (0, 0, 255), -1)
    cv2.putText(frame, "Approaching", (legend_x + 25, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.rectangle(frame, (legend_x, 114), (legend_x + 20, 127), (255, 0, 0), -1)
    cv2.putText(frame, "Normal", (legend_x + 25, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

    cv2.imshow("Aircraft Restricted Zone Monitoring", frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "restricted_zone_events.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Track ID", "Event", "Zone", "Start Time", "End Time", "Duration (sec)"])
    for event in event_log:
        writer.writerow([event["track_id"], event["event"], event["zone"],
                        event["start_time"], event["end_time"], event["duration"]])

print(f"Event log saved to: {csv_path}")
print(f"Total events logged: {len(event_log)}")
