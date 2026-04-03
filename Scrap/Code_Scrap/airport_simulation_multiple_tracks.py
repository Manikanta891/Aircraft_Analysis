#This code create a simulation of an airport with multiple runways and a taxiway, where multiple aircraft move simultaneously.
#This is code is waste

import cv2
import numpy as np

# -----------------------------
# STEP 1: CREATE AIRPORT IMAGE
# -----------------------------
h, w = 600, 1000
bg = np.ones((h, w, 3), dtype=np.uint8) * 255

# Runway 1
cv2.rectangle(bg, (100, 200), (900, 260), (50, 50, 50), -1)

# Runway 2
cv2.rectangle(bg, (100, 320), (900, 380), (60, 60, 60), -1)

# Taxiway
cv2.rectangle(bg, (100, 450), (900, 500), (80, 80, 80), -1)

# Vertical track
cv2.rectangle(bg, (450, 100), (500, 550), (70, 70, 70), -1)

# Markings
for i in range(120, 880, 80):
    cv2.rectangle(bg, (i, 225), (i+40, 235), (255,255,255), -1)
    cv2.rectangle(bg, (i, 345), (i+40, 355), (255,255,255), -1)

cv2.imwrite("airport.png", bg)

# -----------------------------
# STEP 2: LOAD AIRCRAFT PNG
# -----------------------------
plane = cv2.imread("aeroplane.png", cv2.IMREAD_UNCHANGED)

if plane is None:
    print("❌ ERROR: aeroplane.png not found")
    exit()

plane = cv2.resize(plane, (70, 70))

if plane.shape[2] == 4:
    plane_rgb = plane[:, :, :3]
    plane_alpha = plane[:, :, 3] / 255.0
else:
    plane_rgb = plane
    plane_alpha = np.ones((plane.shape[0], plane.shape[1]))

# -----------------------------
# OVERLAY FUNCTION
# -----------------------------
def overlay(bg, fg, alpha, x, y):
    h, w = fg.shape[:2]
    if y + h > bg.shape[0] or x + w > bg.shape[1]:
        return bg

    for c in range(3):
        bg[y:y+h, x:x+w, c] = (
            alpha * fg[:, :, c] +
            (1 - alpha) * bg[y:y+h, x:x+w, c]
        )
    return bg

# -----------------------------
# VIDEO WRITER
# -----------------------------
out = cv2.VideoWriter(
    "simulation.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    20,
    (w, h)
)

# -----------------------------
# MULTI AIRCRAFT INITIALIZATION
# -----------------------------
planes = [
    {"x": 50,  "y": 210, "dx": 4,  "dy": 0},   # Plane 1 → right
    {"x": 900, "y": 330, "dx": -4, "dy": 0},   # Plane 2 → left
    {"x": 470, "y": 550, "dx": 0,  "dy": -3},  # Plane 3 → up
]

# -----------------------------
# SIMULATION LOOP
# -----------------------------
for i in range(300):

    frame = bg.copy()

    for p in planes:
        x, y = int(p["x"]), int(p["y"])

        # Draw aircraft
        frame = overlay(frame, plane_rgb, plane_alpha, x, y)

        # Update position
        p["x"] += p["dx"]
        p["y"] += p["dy"]

        # Boundary reset (loop movement)
        if p["x"] > 1000: p["x"] = -50
        if p["x"] < -50: p["x"] = 1000
        if p["y"] < -50: p["y"] = 600

    out.write(frame)

    cv2.imshow("Simulation", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

out.release()
cv2.destroyAllWindows()

print("✅ Multi-aircraft simulation saved")