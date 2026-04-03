# Aircraft Analysis Project

## Setup Instructions

### 1. Install Python 3.10 using winget
Open PowerShell as Administrator and run:

```powershell
winget install --id Python.Python.3.10 -e
```

Then verify the installation:

```powershell
py -3.10 --version
```

If `py` is not available, verify with:

```powershell
python --version
```

### 2. Create a Python 3.10 virtual environment
From the repository root:

```powershell
py -3.10 -m venv aircraft_env310
```

Activate the environment:

- PowerShell:
  ```powershell
  .\aircraft_env310\Scripts\Activate.ps1
  ```
- CMD:
  ```cmd
  .\aircraft_env310\Scripts\activate
  ```

### 3. Install dependencies
From the repository root with the environment active:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Note: There is also a `Code/requirements.txt`, but the main dependency list is in the root `requirements.txt`.

## Folder Overview

- `Airport_Simulator/`
  - Contains airport simulation and runway/terminal monitoring scripts.
  - Includes collision detection code, runway and terminal JSON configs, sample videos, and output artifacts.

- `Code/`
  - Contains the Streamlit app and video tracking code.
  - Includes `app.py`, `Deepsort.py`, and supporting assets such as `temp.jpg` if generated during execution.

- `Models/`
  - Stores the YOLO aircraft detection model file: `aircraft_detector_v8.pt`.

- `Media/`
  - Stores static images used by the simulator, including runway backgrounds and aircraft sprites.

- `Simulation_Videos/`
  - Contains source videos used by tracking, occupancy, and collision detection scripts.

- `aircraft_env310/`
  - Python 3.10 virtual environment folder created by the user.

## Primary Code Files and Execution Order

### 1. `Code/app.py` (Streamlit)
This is a simple Streamlit web application for image-based aircraft detection.

What it does:
- Loads a YOLO model from `Models/aircraft_detector_v8.pt`.
- Accepts an uploaded `jpg` or `png` image.
- Detects aircraft and draws bounding boxes.
- Calculates:
  - aircraft count
  - density status (`Low`, `Medium`, `High`)
  - activity level (`No Activity`, `Low`, `Moderate`, `High`)

How to run:

```powershell
cd Code
streamlit run app.py
```

Open the Streamlit URL shown in the terminal and upload an image.

### 2. `Code/Deepsort.py`
This script performs video-based aircraft detection with object tracking.

What it does:
- Loads the YOLO model and a DeepSort tracker.
- Reads a source video from `Simulation_Videos/simulation-3.mp4`.
- Performs detection on each frame.
- Tracks each aircraft with persistent IDs.
- Draws detection and tracking boxes on the frame.
- Logs entry/exit times and status to `aircraft_log.csv`.

How to run:

```powershell
cd Code
python Deepsort.py
```

### 3. `Airport_Simulator/terminal_occupancy.py`
This script monitors terminal occupancy based on aircraft detection.

What it does:
- Loads terminal area definitions from `Airport_Simulator/terminals.json`.
- Loads the YOLO aircraft model.
- Reads input video from `Simulation_Videos/simulation-4.mp4`.
- Detects aircraft and computes the center point for each detection.
- Marks terminal boxes as `Occupied` when an aircraft center is inside the terminal zone.
- Displays live video with aircraft boxes and a terminal status panel.

How to run:

```powershell
cd Airport_Simulator
python terminal_occupancy.py
```

### 4. `Airport_Simulator/collision_detection_2.py` (final collision detection)
This is the final version of the collision detection system.

What it does:
- Loads runway definitions from `Airport_Simulator/runways.json`.
- Loads the YOLO model and DeepSort tracker.
- Reads input video from `Simulation_Videos/runway_simulation.mp4`.
- Tracks aircraft positions and trajectories.
- Detects when aircraft are on any configured runway.
- Computes trajectory vectors, relative angles, and runway collision risk.
- Flags `HIGH_RISK`, `BUILDING`, or `SAFE` conditions.
- Displays runway overlays, risk annotations, and a legend.
- Writes output video to `Airport_Simulator/runway_collision_output.avi`.

How to run:

```powershell
cd Airport_Simulator
python collision_detection_2.py
```

> Use `collision_detection_2.py` as the final collision detection implementation. Other collision scripts are earlier versions.

### 5. `Airport_Simulator/full_control.py`
This is an interactive aircraft movement simulator.

What it does:
- Loads a runway background from `Media/runway.png`.
- Loads an aircraft sprite from `Media/aeroplane.png`.
- Prompts for the number of aircraft and per-aircraft start delays.
- Allows interactive path drawing by clicking on the runway image.
- Uses keyboard controls to adjust angle, speed, and size while recording each aircraft path.
- Saves the recorded path data to `simulation_path.json`.
- Simulates aircraft movement along the drawn paths.
- Generates `simulation.mp4` with the animated aircraft.
How to run:

Look for instructions in the terminal

```powershell
cd Airport_Simulator
python full_control.py
```

## Notes and Tips

- Ensure the virtual environment is activated before running any scripts.
- Confirm the model file exists at `Models/aircraft_detector_v8.pt`.
- Confirm the required input videos exist under `Simulation_Videos/`.
- Use `ESC` to stop OpenCV windows during video or interactive simulation.

## Recommended Execution Flow

1. Install Python 3.10 and create the environment.
2. Install dependencies from `requirements.txt`.
3. Run `Code/app.py` for image-level aircraft detection.
4. Run `Code/Deepsort.py` for video tracking and logging.
5. Run `Airport_Simulator/terminal_occupancy.py` for terminal occupancy monitoring.
6. Run `Airport_Simulator/collision_detection_2.py` for runway collision risk detection.
7. Run `Airport_Simulator/full_control.py` for the interactive simulation and aircraft movement control.

---

## Troubleshooting

- If you receive a NumPy/Scikit-learn binary incompatibility error, re-install dependencies after creating the correct Python 3.10 environment.
- If the Streamlit app fails to import `ultralytics`, verify the environment and dependency installation.
- If OpenCV windows fail to open, make sure `python` is running from the activated environment.
