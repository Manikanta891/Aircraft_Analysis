import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

st.set_page_config(page_title="Aircraft Monitoring System", layout="wide")

script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(script_dir).resolve().parent
MODEL_PATH = ROOT_DIR / "Models" / "aircraft_detector_v8.pt"
RUNWAY_CONFIG_PATH = os.path.join(script_dir, "runways.json")

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolov8n.pt"

TRAJECTORY_HISTORY_SIZE = 8
SPEED_THRESHOLD = 3.0
ANGLE_TOWARD_THRESHOLD = 30.0
ANGLE_AWAY_THRESHOLD = 60.0
MIN_CONSECUTIVE_FRAMES = 4
MIN_TRAJECTORY_FRAMES = 5


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


class CollisionDetector:
    def __init__(self, runways_list: List[Dict]):
        self.tracks: Dict[int, AircraftTrack] = {}
        self.runways = runways_list
        self.consecutive_frames: Dict[Tuple[int, int], int] = {}
        self.angle_history: Dict[Tuple[int, int], deque] = {}

    def is_point_in_runway(self, px: float, py: float, runway: dict) -> bool:
        x1 = min(runway['x1'], runway['x2'])
        x2 = max(runway['x1'], runway['x2'])
        y1 = min(runway['y1'], runway['y2'])
        y2 = max(runway['y1'], runway['y2'])
        return x1 <= px <= x2 and y1 <= py <= y2

    def is_aircraft_on_any_runway(self, cx: float, cy: float) -> Tuple[bool, Optional[Dict]]:
        for runway in self.runways:
            if self.is_point_in_runway(cx, cy, runway):
                return True, runway
        return False, None

    def update_track(self, track_id: int, cx: float, cy: float, frame_num: int):
        if track_id not in self.tracks:
            self.tracks[track_id] = AircraftTrack(track_id=track_id)
        on_runway, runway = self.is_aircraft_on_any_runway(cx, cy)
        self.tracks[track_id].on_runway = on_runway
        self.tracks[track_id].runway_id = self.runways.index(runway) + 1 if runway else None
        self.tracks[track_id].positions.append((cx, cy))
        self.tracks[track_id].timestamps.append(frame_num)

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
            return "SAFE", 0.0, "Different runways"
        pair_key = (id_a, id_b)
        reverse_key = (id_b, id_a)
        angle_ab_val, valid_ab = self.compute_pair_angle(id_a, id_b)
        angle_ba_val, valid_ba = self.compute_pair_angle(id_b, id_a)
        if not valid_ab and not valid_ba:
            return "SAFE", 0.0, "Insufficient data"
        risk_a_toward_b = False
        risk_b_toward_a = False
        angle_ab = angle_ab_val if angle_ab_val is not None else 0.0
        angle_ba = angle_ba_val if angle_ba_val is not None else 0.0
        if valid_ab and angle_ab_val is not None and angle_ab_val < ANGLE_TOWARD_THRESHOLD:
            risk_b_toward_a = True
        if valid_ba and angle_ba_val is not None and angle_ba_val < ANGLE_TOWARD_THRESHOLD:
            risk_a_toward_b = True
        if risk_a_toward_b:
            if pair_key not in self.consecutive_frames:
                self.consecutive_frames[pair_key] = 0
            self.consecutive_frames[pair_key] += 1
            if reverse_key in self.consecutive_frames:
                self.consecutive_frames[reverse_key] = max(0, self.consecutive_frames[reverse_key] - 1)
            count = self.consecutive_frames[pair_key]
            if count >= MIN_CONSECUTIVE_FRAMES:
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
                return "HIGH_RISK", angle_ba, f"B->A: {angle_ba:.1f} deg"
            return "BUILDING", angle_ba, f"B->A: {angle_ba:.1f} deg ({count}/{MIN_CONSECUTIVE_FRAMES})"
        else:
            if pair_key in self.consecutive_frames:
                self.consecutive_frames[pair_key] = max(0, self.consecutive_frames[pair_key] - 1)
            if reverse_key in self.consecutive_frames:
                self.consecutive_frames[reverse_key] = max(0, self.consecutive_frames[reverse_key] - 1)
            angle = angle_ab if valid_ab else angle_ba
            return "SAFE", angle, f"Angle: {angle:.1f} deg"

    def analyze_all_pairs(self) -> List[Tuple[int, int, str, float, str]]:
        collision_risks = []
        track_ids = list(self.tracks.keys())
        for i, id_a in enumerate(track_ids):
            for id_b in track_ids[i + 1:]:
                risk, angle, reason = self.analyze_collision_risk(id_a, id_b)
                collision_risks.append((id_a, id_b, risk, angle, reason))
        return collision_risks

    def cleanup_tracks(self, active_ids: set):
        ids_to_remove = [tid for tid in self.tracks.keys() if tid not in active_ids]
        for tid in ids_to_remove:
            if tid in self.tracks:
                del self.tracks[tid]


def load_runways(filepath: str) -> List[Dict]:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return []


def draw_runways(img: np.ndarray, runways_list: List[Dict], active_runways: set = None):
    for i, runway in enumerate(runways_list):
        x1, y1 = int(runway['x1']), int(runway['y1'])
        x2, y2 = int(runway['x2']), int(runway['y2'])
        color = (0, 165, 255)
        if active_runways and i in active_runways:
            color = (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"RWY {i + 1}"
        cv2.putText(img, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_analytics_overlay(frame: np.ndarray, stats: dict, opacity: float = 0.6):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    panel_h = 200
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    frame[:panel_h, :] = cv2.addWeighted(frame[:panel_h, :], 1 - opacity, overlay[:panel_h, :], opacity, 0)
    
    cv2.putText(frame, "AIRCRAFT ANALYTICS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Total Aircraft: {stats.get('total_aircraft', 0)}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"On Runway: {stats.get('on_runway', 0)} | Off Runway: {stats.get('off_runway', 0)}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    risk_color = (0, 0, 255) if stats.get('collision_warnings', 0) > 0 else (0, 255, 0)
    cv2.putText(frame, f"Collision Warnings: {stats.get('collision_warnings', 0)}", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 1)
    cv2.putText(frame, f"High Risk Pairs: {stats.get('high_risk_pairs', 0)}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"Monitoring: {stats.get('building_count', 0)}", (10, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    cv2.putText(frame, f"Frame: {stats.get('frame', 0)}", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    legend_y = h - 80
    cv2.rectangle(frame, (w - 180, legend_y - 10), (w - 10, legend_y + 60), (20, 20, 20), -1)
    cv2.putText(frame, "LEGEND", (w - 170, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(frame, (w - 170, legend_y + 15), (w - 150, legend_y + 30), (0, 0, 255), -1)
    cv2.putText(frame, "Risk", (w - 145, legend_y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.rectangle(frame, (w - 170, legend_y + 35), (w - 150, legend_y + 50), (0, 255, 0), -1)
    cv2.putText(frame, "Safe", (w - 145, legend_y + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)


@st.cache_resource
def load_model():
    try:
        return YOLO(str(MODEL_PATH))
    except:
        return YOLO("yolov8n.pt")


def process_video(video_path: str, mode: str, runways: List[Dict], opacity: float = 0.6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_frames = []
    model = load_model()
    tracker = DeepSort(max_age=30, n_init=3)
    detector = CollisionDetector(runways)
    
    frame_count = 0
    collision_warnings = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_detections = []
        
        if mode in ["detection", "analytics"]:
            results = model(frame, conf=0.05, imgsz=1280)[0]
            
            if results.boxes is not None:
                for r in results.boxes.data:
                    x1, y1, x2, y2, conf, cls = r.tolist()
                    w, h = x2 - x1, y2 - y1
                    current_detections.append(([x1, y1, w, h], conf, 'aircraft'))
            
            tracks = tracker.update_tracks(current_detections, frame=frame)
            
            confirmed_tracks = []
            active_ids = set()
            
            for track in tracks:
                if not track.is_confirmed() or track.hits < 3:
                    continue
                l, t, r, b = track.to_ltrb()
                cx, cy = (l + r) / 2, (t + b) / 2
                track_id = track.track_id
                active_ids.add(track_id)
                
                detector.update_track(track_id, cx, cy, frame_count)
                
                confirmed_tracks.append({
                    'id': track_id, 'cx': cx, 'cy': cy,
                    'l': l, 't': t, 'r': r, 'b': b
                })
            
            detector.cleanup_tracks(active_ids)
            
            if mode == "detection":
                draw_runways(frame, runways)
                
                for track in confirmed_tracks:
                    tid = track['id']
                    x1, y1, x2, y2 = int(track['l']), int(track['t']), int(track['r']), int(track['b'])
                    track_data = detector.tracks.get(tid)
                    
                    if track_data and track_data.on_runway:
                        color = (0, 255, 255)
                        status = "ON RWY"
                    else:
                        color = (0, 255, 0)
                        status = "OFF RWY"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{tid} | {status}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    if track_data and track_data.runway_id:
                        cv2.putText(frame, f"RWY:{track_data.runway_id}", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            elif mode == "analytics":
                collision_analysis = detector.analyze_all_pairs()
                
                risky_ids = set()
                building_ids = set()
                active_runways = set()
                
                for id_a, id_b, risk, angle, reason in collision_analysis:
                    if risk == "HIGH_RISK":
                        risky_ids.add(id_a)
                        risky_ids.add(id_b)
                        collision_warnings += 1
                        track_a = detector.tracks.get(id_a)
                        track_b = detector.tracks.get(id_b)
                        if track_a and track_a.runway_id:
                            active_runways.add(track_a.runway_id - 1)
                        if track_b and track_b.runway_id:
                            active_runways.add(track_b.runway_id - 1)
                    elif risk == "BUILDING":
                        building_ids.add(id_a)
                        building_ids.add(id_b)
                
                draw_runways(frame, runways, active_runways)
                
                for track in confirmed_tracks:
                    tid = track['id']
                    x1, y1, x2, y2 = int(track['l']), int(track['t']), int(track['r']), int(track['b'])
                    track_data = detector.tracks.get(tid)
                    
                    if tid in risky_ids:
                        color = (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                        cv2.putText(frame, f"ID:{tid} COLLISION RISK!", (x1, y1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    elif tid in building_ids:
                        color = (0, 165, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(frame, f"ID:{tid} MONITORING", (x1, y1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    elif track_data and track_data.on_runway:
                        color = (0, 255, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID:{tid} | ON RWY", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        if track_data.runway_id:
                            cv2.putText(frame, f"RWY:{track_data.runway_id}", (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    else:
                        color = (100, 100, 100)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                        cv2.putText(frame, f"ID:{tid} | OFF RWY", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                for id_a, id_b, risk, _, _ in collision_analysis:
                    if risk == "HIGH_RISK":
                        track_a = next((t for t in confirmed_tracks if t['id'] == id_a), None)
                        track_b = next((t for t in confirmed_tracks if t['id'] == id_b), None)
                        if track_a and track_b:
                            cv2.line(frame, (int(track_a['cx']), int(track_a['cy'])),
                                    (int(track_b['cx']), int(track_b['cy'])), (0, 0, 255), 3)
                
                stats = {
                    'total_aircraft': len(confirmed_tracks),
                    'on_runway': len([t for t in confirmed_tracks if detector.tracks.get(t['id']) and detector.tracks.get(t['id']).on_runway]),
                    'off_runway': len([t for t in confirmed_tracks if detector.tracks.get(t['id']) and not detector.tracks.get(t['id']).on_runway]),
                    'collision_warnings': collision_warnings,
                    'high_risk_pairs': len(risky_ids) // 2,
                    'building_count': len(building_ids),
                    'frame': frame_count
                }
                draw_analytics_overlay(frame, stats, opacity)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_frames.append(frame_rgb)
        
        progress = int((frame_count / total_frames) * 100)
        progress_bar.progress(min(progress, 100))
        status_text.text(f"Processing Frame {frame_count}/{total_frames}")
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return output_frames


def main():
    st.title("Aircraft Monitoring System")
    st.markdown("### Real-time Aircraft Detection, Tracking & Collision Risk Analysis")
    
    runways = load_runways(RUNWAY_CONFIG_PATH)
    
    with st.sidebar:
        st.header("Settings")
        
        uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        
        st.subheader("Display Mode")
        mode = st.radio(
            "Select Mode",
            options=["raw", "detection", "analytics"],
            format_func=lambda x: {
                "raw": "1. Raw Footage",
                "detection": "2. Detection & Tracking",
                "analytics": "3. Analytics Dashboard"
            }[x],
            horizontal=True
        )
        
        st.subheader("Runway Configuration")
        st.write(f"Loaded: {len(runways)} runway(s)")
        if runways:
            for i, r in enumerate(runways):
                st.write(f"  RWY {i+1}: ({r['x1']},{r['y1']}) - ({r['x2']},{r['y2']})")
        
        st.subheader("Analytics Settings")
        opacity = st.slider("Overlay Opacity", 0.0, 1.0, 0.6, 0.1)
        
        st.subheader("Detection Parameters")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.05, 0.01)
        speed_thresh = st.slider("Speed Threshold", 1.0, 10.0, 3.0, 0.5)
        angle_thresh = st.slider("Angle Threshold (deg)", 10.0, 60.0, 30.0, 5.0)
        consecutive = st.slider("Min Consecutive Frames", 1, 10, 4, 1)
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        video_path = tfile.name
        
        if 'processed_frames' not in st.session_state:
            st.session_state.processed_frames = None
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Video Output")
            video_placeholder = st.empty()
        
        with col2:
            st.subheader("Statistics")
            stats_placeholder = st.empty()
        
        if st.button("Process Video", type="primary"):
            with st.spinner("Processing video..."):
                frames = process_video(video_path, mode, runways, opacity)
                
                if frames:
                    st.session_state.processed_frames = frames
                    st.session_state.current_frame = 0
        
        if st.session_state.processed_frames is not None:
            frames = st.session_state.processed_frames
            
            for i, frame in enumerate(frames):
                video_placeholder.image(frame, channels="RGB", use_container_width=True)
                
                stats_placeholder.info(f"Frame {i+1}/{len(frames)} | Mode: {mode}")
        
        os.unlink(video_path)
    
    else:
        st.info("Please upload a video file to begin analysis")
        
        st.subheader("Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Raw Footage")
            st.markdown("View original video without any processing")
        
        with col2:
            st.markdown("### Detection & Tracking")
            st.markdown("- YOLO object detection\n- DeepSort tracking\n- Runway identification")
        
        with col3:
            st.markdown("### Analytics Dashboard")
            st.markdown("- Collision risk detection\n- Trajectory analysis\n- Real-time statistics")


if __name__ == "__main__":
    main()
