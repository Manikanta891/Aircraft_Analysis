# Aircraft Analysis Project

A smart airport monitoring system using computer vision and deep learning to detect, track, and monitor aircraft in real-time.

## Features

- **Aircraft Detection** - YOLO-based object detection
- **Object Tracking** - Persistent tracking with DeepSort
- **Runway Collision Detection** - Trajectory-based risk assessment
- **Terminal Occupancy** - Gate availability monitoring
- **Parking Slot Management** - Slot occupancy tracking
- **Restricted Zone Monitoring** - Unauthorized entry detection
- **Real-time Dashboards** - Web-based monitoring interfaces

## Quick Start

```powershell
# Install Python 3.10
winget install --id Python.Python.3.10 -e

# Create virtual environment
py -3.10 -m venv aircraft_env310

# Activate
.\aircraft_env310\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Project

### Web Dashboards
```powershell
cd Dashboard-1 && python app.py   # Runway (Port 5001)
cd Dashboard-2 && python app.py   # Restricted Zones (Port 5002)
cd Dashboard-3 && python app.py   # Parking (Port 5003)
cd Dashboard-4 && python app.py   # Terminal (Port 5004)
```

### Standalone Scripts
```powershell
cd Code\Terminal && python terminal_occupancy.py
cd Code\Runway && python collision_detection-2.py
cd Code\Restricted && python restricted_zone.py
cd Code\Parking && python parking_occupancy.py
```

## Requirements

- Python 3.10
- YOLOv11 model (included)
- OpenCV, Flask, DeepSort

## Project Structure

```
Aircraft_Analysis/
├── Code/           # Detection & analysis scripts
├── Dashboard-{1-4}/ # Web dashboards
├── Models/         # ML models
├── Media/         # Static assets
└── Simulation_Videos/ # Sample videos
```

## License

Final Year Project
