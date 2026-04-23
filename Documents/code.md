# SMART AIRPORT MONITORING SYSTEM - KEY CODE

---

## SECTION 1: YOLO11 TRAINING PIPELINE

### 1.1 Dataset Configuration (`data.yaml`)
```python
# data.yaml — dataset configuration

train: /content/drive/MyDrive/AircraftProject/dataset_satellite/train/images
val:   /content/drive/MyDrive/AircraftProject/dataset_satellite/valid/images

nc: 1
names: ['aircraft']
```

### 1.2 Environment Setup (Colab)
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install Ultralytics
!pip install -U ultralytics

# Check GPU
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### 1.3 YOLO11 Training Code (Main Model)
```python
from ultralytics import YOLO

model = YOLO('yolo11m.pt')

results = model.train(
    data='/content/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,

    name='aircraft_yolo11',
    project='/content/drive/MyDrive/AircraftProject/runs',

    patience=20,

    augment=True,
    mosaic=1.0,
    degrees=45.0,
    flipud=0.5,
    fliplr=0.5,

    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,

    conf=0.25,
    iou=0.5,

    save=True,
    save_period=10
)

print("Training completed")
```

### 1.4 Model Evaluation Code
```python
from ultralytics import YOLO

model = YOLO('/content/drive/MyDrive/AircraftProject/runs/aircraft_yolo11/weights/best.pt')

metrics = model.val(
    data='/content/data.yaml',
    imgsz=640,
    batch=16
)

print("mAP50:", metrics.box.map50)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
```

### 1.5 Inference Code (Real Usage)
```python
from ultralytics import YOLO

model = YOLO('C:/AircraftProject/models/aircraft_yolo11_best.pt')

results = model(
    'test_image.jpg',
    conf=0.40
)

for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    print(f"Aircraft detected at ({x1},{y1},{x2},{y2}) with confidence {conf:.2f}")

print("Total Aircraft:", len(results[0].boxes))
```

### 1.6 Switching Between Models
```python
# Accuracy mode
MODEL_PATH = '../models/aircraft_yolo11_best.pt'

# Speed mode
MODEL_PATH = '../models/aircraft_yolo26_best.pt'
```

### 1.7 Save Model Code
```python
import shutil

shutil.copy(
    '/content/drive/MyDrive/AircraftProject/runs/aircraft_yolo11/weights/best.pt',
    '/content/drive/MyDrive/AircraftProject/models/aircraft_yolo11_best.pt'
)
```

---

## SECTION 2: COMMON DETECTION & TRACKING CODE

### 2.1 YOLO Aircraft Detection
```python
# Initialize model and detect aircraft
model = YOLO(str(MODEL_PATH))
results = model(frame, conf=0.25, imgsz=640)[0]

# Extract bounding boxes
current_detections = []
if results.boxes is not None:
    for r in results.boxes.data:
        x1, y1, x2, y2, conf, cls = r.tolist()
        w = x2 - x1
        h = y2 - y1
        current_detections.append(([x1, y1, w, h], conf, 'aircraft'))
```

### 2.2 DeepSort Object Tracking
```python
# Initialize tracker
tracker = DeepSort(max_age=30, n_init=3)

# Update tracks with detections
tracks = tracker.update_tracks(current_detections, frame=frame)

# Process confirmed tracks
for track in tracks:
    if not track.is_confirmed() or track.hits < 3:
        continue
    l, t, r, b = track.to_ltrb()
    cx = (l + r) / 2  # Center X
    cy = (t + b) / 2  # Center Y
    track_id = track.track_id
```

---

## SECTION 3: DASHBOARD-1 - RUNWAY COLLISION DETECTION

### 3.1 Aircraft Track Dataclass
```python
from collections import deque
from dataclasses import dataclass, field

@dataclass
class AircraftTrack:
    track_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=5))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=5))
    on_runway: bool = False
    runway_id: Optional[int] = None
    
    def get_trajectory_vector(self):
        current = self.positions[-1]
        oldest = self.positions[0]
        mvx = current[0] - oldest[0]
        mvy = current[1] - oldest[1]
        speed = np.sqrt(mvx**2 + mvy**2)
        return mvx, mvy, speed
```

### 3.2 Collision Detection Algorithm
```python
def compute_pair_angle(self, moving_id: int, target_id: int):
    moving = self.tracks.get(moving_id)
    target = self.tracks.get(target_id)
    
    trajectory = moving.get_trajectory_vector()
    mvx, mvy, speed = trajectory
    
    current = moving.get_current_position()
    dx = target_pos[0] - current[0]
    dy = target_pos[1] - current[1]
    
    # Calculate angle using dot product
    dot = (mvx * dx + mvy * dy) / (speed * dist)
    angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg
```

### 3.3 Collision Risk Analysis
```python
def analyze_collision_risk(self, id_a: int, id_b: int):
    track_a = self.tracks.get(id_a)
    track_b = self.tracks.get(id_b)
    
    # Check if both aircraft are on same runway
    if track_a.runway_id != track_b.runway_id:
        return "SAFE", 0.0, "Different runways"
    
    # Analyze trajectory angles
    angle_ab = self.compute_pair_angle(id_a, id_b)
    
    if angle_ab < 30:  # Moving toward each other
        self.consecutive_frames[(id_a, id_b)] += 1
        if self.consecutive_frames[(id_a, id_b)] >= 4:
            return "HIGH_RISK", angle_ab, "Collision imminent"
        return "BUILDING", angle_ab, "Risk building"
    elif angle_ab > 60:  # Moving away
        return "SAFE", angle_ab, "Moving apart"
```

### 3.4 API Endpoints (Flask Routes)
```python
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/data')
def get_data():
    with frame_lock:
        return jsonify(current_data)

@app.route('/api/start', methods=['POST'])
def start_stream():
    processing_thread = threading.Thread(target=process_video)
    processing_thread.start()
    return jsonify({"status": "started"})
```

---

## SECTION 4: DASHBOARD-2 - RESTRICTED ZONE MONITORING

### 4.1 Zone Status Check
```python
def check_restricted_zone(self, cx: float, cy: float):
    for zone in self.zones:
        restricted = zone.get('restricted', zone)
        x1, y1 = restricted['x1'], restricted['y1']
        x2, y2 = restricted['x2'], restricted['y2']
        if min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2):
            return True, zone
    return False, None
```

### 4.2 Severity Analysis (Time-based)
```python
def analyze_severity(self, track: AircraftTrack):
    if not track.in_restricted_zone:
        return "NORMAL", 0.0
    
    time_inside = time.time() - track.entry_time
    
    if time_inside > 10:  # Critical: 10+ seconds
        return "CRITICAL", time_inside
    elif time_inside > 5:  # Warning: 5+ seconds
        return "WARNING", time_inside
    return "NORMAL", time_inside
```

---

## SECTION 5: DASHBOARD-3 - PARKING SLOT MONITORING

### 5.1 Parking Slot Dataclass
```python
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
```

### 5.2 Aircraft to Slot Assignment
```python
def assign_aircraft_to_slots(self, aircraft_positions):
    for slot in self.slots:
        for aid, (cx, cy) in aircraft_positions.items():
            if slot.contains_point(cx, cy):
                if slot.aircraft_id is None:
                    slot.aircraft_id = aid
                    slot.entry_time = time.time()
                    slot.status = "OCCUPIED"
                    self.add_log(f"Aircraft {aid} parked at {slot.id}", "INFO")
                    break
```

### 5.3 Long Stay Detection
```python
def analyze_slots(self):
    for slot in self.slots:
        if slot.aircraft_id is not None:
            duration = time.time() - slot.entry_time
            
            if duration > 120:  # Critical: 2+ minutes
                self.add_log(f"CRITICAL: Aircraft at {slot.id} exceeded 120s", "CRITICAL")
            elif duration > 60:  # Warning: 1+ minute
                self.add_log(f"WARNING: Aircraft at {slot.id} staying long", "WARNING")
```

---

## SECTION 6: DASHBOARD-4 - TERMINAL GATE MONITORING

### 6.1 Terminal Gate Dataclass
```python
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
```

### 6.2 Gate Occupancy Update
```python
def update_aircraft_positions(self, aircraft_positions):
    for aid, (cx, cy) in aircraft_positions.items():
        for gate in self.gates:
            if gate.contains_point(cx, cy):
                if gate.aircraft_id is None:
                    gate.aircraft_id = aid
                    gate.entry_time = time.time()
                    gate.status = "OCCUPIED"
                    self.add_log(f"Aircraft {aid} docked at gate {gate.id}", "INFO")
    
    # Check for departures
    for gate in self.gates:
        if gate.aircraft_id not in aircraft_positions:
            self.add_log(f"Aircraft departed gate {gate.id}", "INFO")
            gate.aircraft_id = None
            gate.status = "FREE"
```

---

## SECTION 7: FRONTEND - Real-time Updates

### 7.1 Start Stream & Fetch Data
```javascript
function startStream() {
    fetch('/api/start', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            document.getElementById('videoFeed').src = '/video_feed';
            streamActive = true;
            setInterval(updateData, 1500);  // Poll every 1.5s
        });
}

function updateData() {
    fetch('/api/data')
        .then(res => res.json())
        .then(data => {
            updateRunwayStatus(data.runway_status);
            updateAircraftTable(data.aircraft);
            updateAlerts(data.alerts);
        });
}
```

### 7.2 Update UI from JSON Data
```javascript
function updateRunwayStatus(status) {
    const grid = document.getElementById('runwayGrid');
    grid.innerHTML = Object.entries(status).map(([id, data]) => `
        <div class="runway-card ${data.status}">
            <div class="runway-id">${id}</div>
            <div class="runway-status">${data.status}</div>
            <div class="runway-info">
                Aircraft: ${data.aircraft_count}<br>
                Utilization: ${data.utilization}
            </div>
        </div>
    `).join('');
}
```

---

## SECTION 8: SYSTEM ARCHITECTURE

### Data Flow Diagram
```
[Video Input] → [YOLO Detector] → [DeepSort Tracker] → [Domain Analysis]
                                                              ↓
                    [Flask API] ← [Shared State (Lock)] ← [Thread]
                         ↓
                  [MJPEG Stream] + [JSON API]
                         ↓
                  [Browser Dashboard]
```

### Thread-Safe State Sharing
```python
frame_lock = threading.Lock()
current_frame = None
current_data = {}

def process_video():
    # ... process frame ...
    with frame_lock:
        _, buffer = cv2.imencode('.jpg', frame)
        current_frame = buffer.tobytes()
        current_data = {"runway_status": ..., "aircraft": ..., "alerts": ...}
```

---

## SECTION 9: KEY PARAMETERS

| Parameter | Value |
|-----------|-------|
| YOLO Confidence | 0.25 |
| DeepSort max_age | 30 frames |
| DeepSort n_init | 3 confirmations |
| Trajectory History | 5 frames |
| Speed Threshold | 3.0 pixels/frame |
| Collision Angle | <30° toward, >60° away |
| Consecutive Frames | 4 minimum for HIGH_RISK |
| API Poll Interval | 1.5 seconds |
| FPS Target | 30 |

---

