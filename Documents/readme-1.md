# OpenCode - Aircraft Analysis Project

## Project Overview

This is a **Smart Airport Monitoring System** - a Final Year Project that uses computer vision and deep learning to detect, track, and monitor aircraft in airport environments. It provides real-time analysis of runway operations, terminal occupancy, parking slot management, and restricted zone monitoring with collision detection.

---

## Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Detection | YOLOv11 (Ultralytics) | Aircraft object detection |
| Tracking | DeepSort | Persistent aircraft tracking across frames |
| Web Framework | Flask | Web dashboard backend |
| Web UI | HTML/CSS/JavaScript | Dashboard frontend |
| Image Processing | OpenCV | Video frame processing |
| Data Storage | JSON, CSV | Configuration and logs |

---

## Directory Structure

```
Aircraft_Analysis/
├── Code/                              # Core analysis scripts
│   ├── Common/
│   │   └── base_editor.py            # Base class for interactive zone editors
│   ├── Parking/
│   │   ├── parking_occupancy.py      # Parking slot occupancy monitoring
│   │   └── parking_setup.py          # Interactive parking slot editor
│   ├── Restricted/
│   │   ├── restricted_zone.py         # Restricted zone monitoring
│   │   └── restricted_setup.py        # Interactive zone editor
│   ├── Runway/
│   │   ├── runway_designer.py         # Interactive runway zone editor
│   │   ├── collision_detection-1.py  # Collision detection v1
│   │   └── collision_detection-2.py  # Collision detection v2 (FINAL)
│   ├── Terminal/
│   │   ├── terminal_occupancy.py      # Terminal gate occupancy monitoring
│   │   └── terminal_setup.py          # Interactive terminal editor
│   └── Utils/
│       └── box_viewer.py              # Visualize JSON boxes on images
│
├── Airport_Simulator/                 # Airport simulation system
│   ├── full_control.py               # Interactive aircraft path simulator
│   ├── parking.json                  # Parking slot zone definitions
│   ├── terminals.json                # Terminal gate definitions
│   ├── restricted_zones.json         # Restricted zone definitions
│   └── runways.json                  # Runway definitions
│
├── Dashboard-1/                      # Runway Collision Dashboard (Port 5001)
│   ├── app.py                        # Flask app with MJPEG streaming
│   └── templates/dashboard.html      # Dashboard UI
│
├── Dashboard-2/                      # Restricted Zone Dashboard (Port 5002)
│   ├── app.py
│   └── templates/restricted_zone.html
│
├── Dashboard-3/                     # Parking Dashboard (Port 5003)
│   ├── app.py
│   └── templates/parking.html
│
├── Dashboard-4/                      # Terminal Dashboard (Port 5004)
│   ├── app.py
│   └── templates/index.html
│
├── Models/                           # ML Models
│   ├── aircraft_detector_v11.pt      # YOLOv11 aircraft detector (PRIMARY)
│   └── aircraft_detector_v8.pt       # YOLOv8 aircraft detector (LEGACY)
│
├── Media/                            # Static assets
│   ├── runway.png                    # Runway background
│   ├── parking.jpg                  # Parking area image
│   ├── terminal.jpg                 # Terminal area image
│   ├── restricted.jpg               # Restricted area image
│   ├── aeroplane.png                # Aircraft sprite (PNG with alpha)
│   └── airport.jpg                  # Full airport view
│
├── Simulation_Videos/                # Input/Output videos
│   ├── parking_simulation.mp4       # Parking simulation
│   ├── terminal_simulation.mp4       # Terminal simulation
│   ├── runway_simulation.mp4        # Runway simulation
│   ├── runway_single_aircraft.mp4    # Single aircraft runway
│   ├── runway_away.mp4              # Away runway simulation
│   └── restricted_video.mp4         # Restricted zone simulation
│
├── web_app/                          # Advanced Flask web app
│   ├── app.py                       # Image analysis with clustering
│   ├── templates/
│   ├── uploads/
│   └── outputs/
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Original documentation
├── USAGE.md                          # Usage guide
└── Knowledge_Transfer_Document.md   # Detailed project documentation
```

---

## Key Features

### 1. Aircraft Detection (YOLO)
- **Model**: `aircraft_detector_v11.pt` (Ultralytics YOLOv11)
- **Input**: Images or video frames
- **Output**: Bounding boxes with confidence scores
- **Usage**:
```python
from ultralytics import YOLO
model = YOLO("Models/aircraft_detector_v11.pt")
results = model(frame, conf=0.25, imgsz=640)
```

### 2. Object Tracking (DeepSort)
- **Purpose**: Maintain consistent IDs across frames
- **Configuration**:
```python
tracker = DeepSort(max_age=30, n_init=3)
```
- `max_age=30`: Maximum frames to keep lost tracks
- `n_init=3`: Minimum confirmations before showing track

### 3. Collision Detection Algorithm
- Trajectory history tracking (deque with maxlen=5)
- Speed threshold detection (3.0 pixels/frame)
- Angle-based direction analysis (30 deg toward, 60 deg away)
- Consecutive frame confirmation (4 frames minimum)
- Risk levels: SAFE, BUILDING, HIGH_RISK

### 4. Terminal Occupancy Detection
- Point-in-box algorithm for each terminal gate
- Real-time status updates (Occupied/Free)
- Duration tracking for each parked aircraft
- Long-stay warnings and critical alerts

### 5. Parking Slot Monitoring
- JSON-based slot definitions (no transformation)
- Direct overlay on video frames
- Occupancy tracking with duration
- Status panel display

### 6. Restricted Zone Monitoring
- Inner zone: Immediate violation detection
- Outer zone: Warning/approach detection
- Time-based severity levels (5s warning, 10s critical)
- Trajectory prediction for approaching aircraft

---

## Code Conventions Applied

### Path Resolution
- All scripts use `ROOT_DIR` derived from `script_dir` for correct path resolution
- JSON config files loaded from `Airport_Simulator/` directory
- Model files loaded from `Models/` directory
- Videos loaded from `Simulation_Videos/` directory

### No Transformation Logic
- JSON boxes are used directly without scaling/offset
- Box coordinates match video resolution
- If JSON coords don't match video, adjust the JSON file instead

### Render Fix for BaseEditor
- Changed `render(None)` to `render(np.zeros(...))` in `runway_designer.py`
- Ensures image content is properly copied to display buffer

### Confidence Threshold
- Detection confidence set to 0.5 for parking_occupancy.py
- Reduces false positives in occupancy matching

---

## Data Flow Architecture

```
VIDEO INPUT
    |
    v
FRAME READER (cv2.VideoCapture)
    |
    v
YOLO DETECTOR (conf=0.25, imgsz=640)
    |
    v
DEEPSORT TRACKER (max_age=30, n_init=3)
    |
    v
DOMAIN-SPECIFIC ANALYSIS
    ├── Collision Detection
    ├── Terminal Occupancy
    ├── Parking Monitoring
    └── Restricted Zone Monitoring
    |
    v
ANNOTATION LAYER (OpenCV drawing)
    |
    ├── OpenCV Display (cv2.imshow)
    └── MJPEG Encoder (for web streaming)
```

---

## Dashboard Architecture

```
Video Input -> Processing Thread -> Shared State (Lock) -> Web Endpoints
                                          |
                            +--------------+--------------+
                            v                              v
                     MJPEG Stream                  JSON API
                      /video_feed                  /api/data
                            |                              |
                            v                              v
                       <img> tag                   Fetch polling
                            |                              |
                            +--------------+--------------+
                                           v
                                  Browser Dashboard
```

---

## Key Parameters Reference

| Parameter | Value | Location |
|-----------|-------|----------|
| YOLO Confidence | 0.25 | Detection scripts |
| YOLO Image Size | 640 | Detection scripts |
| DeepSort max_age | 30 | All trackers |
| DeepSort n_init | 3 | All trackers |
| Trajectory History | 5 frames | Collision detection |
| Speed Threshold | 3.0 px/frame | Collision detection |
| Angle Toward | < 30 deg | Collision detection |
| Angle Away | > 60 deg | Collision detection |
| Min Consecutive | 4 frames | Collision detection |
| Warning Time | 5 seconds | Restricted zones |
| Critical Time | 10 seconds | Restricted zones |
| FPS Target | 30 | Dashboard apps |
| API Poll Interval | 1.5 seconds | HTML dashboards |

---

## Files Modified in This Session

1. **Dashboard-3/app.py** - Removed slot transformation logic
2. **Code/Parking/parking_occupancy.py** - Removed scale_boxes, use raw JSON coords, draw boxes on video, increased confidence to 0.5
3. **Code/Terminal/terminal_occupancy.py** - Removed scale_boxes, use raw JSON coords, draw boxes on video
4. **Code/Runway/runway_designer.py** - Fixed render() call to pass display buffer

---

## Common Patterns

### Loading JSON Config
```python
script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(script_dir).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "Airport_Simulator" / "parking.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
boxes = config["terminal_boxes"]
```

### Drawing Boxes on Video
```python
for i, box in enumerate(terminal_boxes):
    x, y, w, h = box
    color = (0, 255, 0) if status == "Free" else (0, 165, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"P{i+1}", (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
```

### Point-in-Box Detection
```python
def point_in_box(point, box):
    x, y = point
    bx, by, bw, bh = box
    return bx <= x <= bx + bw and by <= y <= by + bh
```

---

## Troubleshooting

### Issue: Image not showing in runway_designer.py
**Fix**: Changed `render(None)` to `render(np.zeros(...))` in the main loop

### Issue: Boxes don't align with video
**Fix**: JSON coordinates must match video resolution. Edit the JSON file to adjust coordinates.

### Issue: Model file not found
**Fix**: Ensure `ROOT_DIR` resolves to project root (contains `Models/` folder)

### Issue: Port already in use
**Fix**:
```powershell
netstat -ano | findstr :5001
taskkill /PID <pid_number> /F
```
