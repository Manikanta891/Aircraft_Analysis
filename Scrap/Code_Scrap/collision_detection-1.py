# This is distance based collision approach
import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

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

DISTANCE_THRESHOLD = 80
SPEED_THRESHOLD = 2
PREDICTION_FRAMES = 10
HISTORY_SIZE = 5

output_path = os.path.join(script_dir, "collision_detection_output.avi")
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (video_w, video_h))

print("\n" + "=" * 60)
print("AIRCRAFT COLLISION RISK DETECTION SYSTEM")
print("=" * 60)
print(f"Video: {VIDEO_PATH}")
print(f"Resolution: {video_w}x{video_h}")
print(f"Distance Threshold: {DISTANCE_THRESHOLD} pixels")
print(f"Speed Threshold: {SPEED_THRESHOLD} pixels/frame")
print(f"Prediction: {PREDICTION_FRAMES} frames ahead")
print("=" * 60)

track_history = {}
track_velocities = {}
frame_count = 0
collision_warnings = 0

def compute_velocity(history):
    if len(history) < 2:
        return 0, 0
    x1, y1 = history[-2]
    x2, y2 = history[-1]
    return x2 - x1, y2 - y1

def compute_speed(vx, vy):
    return np.sqrt(vx**2 + vy**2)

def predict_position(x, y, vx, vy, n_frames):
    return x + vx * n_frames, y + vy * n_frames

def check_collision_risk(pos1, pos2, vel1, vel2, check_future=True):
    x1, y1 = pos1
    x2, y2 = pos2
    vx1, vy1 = vel1
    vx2, vy2 = vel2
    
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    speed1 = compute_speed(vx1, vy1)
    speed2 = compute_speed(vx2, vy2)
    
    is_static1 = speed1 < SPEED_THRESHOLD
    is_static2 = speed2 < SPEED_THRESHOLD
    
    if is_static1 and is_static2:
        return "SAFE", dist, "Both aircraft stationary"
    
    if dist > DISTANCE_THRESHOLD * 2:
        if not check_future:
            return "SAFE", dist, "Too far apart"
        fx1, fy1 = predict_position(x1, y1, vx1, vy1, PREDICTION_FRAMES)
        fx2, fy2 = predict_position(x2, y2, vx2, vy2, PREDICTION_FRAMES)
        future_dist = np.sqrt((fx2 - fx1)**2 + (fy2 - fy1)**2)
        if future_dist > DISTANCE_THRESHOLD:
            return "SAFE", dist, f"Will not collide (future: {int(future_dist)}px)"
    
    rpx = x1 - x2
    rpy = y1 - y2
    rvx = vx1 - vx2
    rvy = vy1 - vy2
    
    dot = rvx * rpx + rvy * rpy
    
    if dot < 0:
        if dist < DISTANCE_THRESHOLD:
            return "HIGH_RISK", dist, "Colliding now!"
        else:
            return "RISK", dist, f"Moving towards (dot={dot:.0f})"
    else:
        return "SAFE", dist, "Moving apart or parallel"

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
    for track in tracks:
        if not track.is_confirmed() or track.hits < 3:
            continue
        
        l, t, r, b = track.to_ltrb()
        cx = int((l + r) / 2)
        cy = int((t + b) / 2)
        track_id = track.track_id
        
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((cx, cy))
        if len(track_history[track_id]) > HISTORY_SIZE:
            track_history[track_id].pop(0)
        
        vx, vy = compute_velocity(track_history[track_id])
        speed = compute_speed(vx, vy)
        
        status = "MOVING" if speed >= SPEED_THRESHOLD else "STATIC"
        
        confirmed_tracks.append({
            'id': track_id,
            'cx': cx,
            'cy': cy,
            'vx': vx,
            'vy': vy,
            'speed': speed,
            'status': status,
            'l': l, 't': t, 'r': r, 'b': b
        })
    
    collision_pairs = []
    risk_details = {}
    
    for i, track1 in enumerate(confirmed_tracks):
        for j, track2 in enumerate(confirmed_tracks):
            if i >= j:
                continue
            
            pos1 = (track1['cx'], track1['cy'])
            pos2 = (track2['cx'], track2['cy'])
            vel1 = (track1['vx'], track1['vy'])
            vel2 = (track2['vx'], track2['vy'])
            
            risk, dist, reason = check_collision_risk(pos1, pos2, vel1, vel2)
            
            if risk in ["HIGH_RISK", "RISK"]:
                collision_pairs.append((track1['id'], track2['id'], risk, dist, reason))
                risk_details[track1['id']] = (risk, reason)
                risk_details[track2['id']] = (risk, reason)
                
                if risk == "HIGH_RISK":
                    collision_warnings += 1
                    print(f"[!] COLLISION ALERT: Aircraft {track1['id']} <-> Aircraft {track2['id']} | Dist: {int(dist)}px | {reason}")
    
    for track in confirmed_tracks:
        tid = track['id']
        x1, y1, x2, y2 = int(track['l']), int(track['t']), int(track['r']), int(track['b'])
        speed = track['speed']
        
        if tid in risk_details:
            risk, reason = risk_details[tid]
            if risk == "HIGH_RISK":
                color = (0, 0, 255)
                thickness = 4
            else:
                color = (0, 100, 255)
                thickness = 3
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"ID:{tid} {risk}!"
            cv2.putText(frame, label, (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Speed: {speed:.1f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        else:
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"ID:{tid} | {track['status']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    for id1, id2, risk, dist, reason in collision_pairs:
        track1 = next((t for t in confirmed_tracks if t['id'] == id1), None)
        track2 = next((t for t in confirmed_tracks if t['id'] == id2), None)
        if track1 and track2:
            x1, y1 = track1['cx'], track1['cy']
            x2, y2 = track2['cx'], track2['cy']
            line_color = (0, 0, 255) if risk == "HIGH_RISK" else (0, 100, 255)
            cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(frame, f"{int(dist)}px", (mid_x, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
    
    panel_h = 140
    cv2.rectangle(frame, (0, 0), (320, panel_h), (0, 0, 0), -1)
    cv2.putText(frame, "AIRCRAFT COLLISION SYSTEM", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f"Tracks: {len(confirmed_tracks)} | Frame: {frame_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    if collision_pairs:
        cv2.putText(frame, f"WARNING: {len(collision_pairs)} RISK(S) DETECTED!", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Status: ALL SAFE", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.putText(frame, f"Total Warnings: {collision_warnings}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.putText(frame, f"Thresh: {DISTANCE_THRESHOLD}px | Pred: {PREDICTION_FRAMES}f", (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    
    cv2.imshow("Aircraft Collision Detection", frame)
    out.write(frame)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("COLLISION DETECTION COMPLETE")
print("=" * 60)
print(f"Total frames processed: {frame_count}")
print(f"Total collision warnings: {collision_warnings}")
print(f"Output saved: {output_path}")
print("=" * 60)
