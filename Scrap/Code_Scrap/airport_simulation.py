# This code is also waste it create single track of aircraft movement, but it is not used in the final app. 
# It was an experiment to create a more complex simulation of an airport with multiple runways and a taxiway, where multiple aircraft move simultaneously. 
# However, it was not integrated into the final Streamlit app as it was deemed unnecessary for the app's functionality.

import cv2
import numpy as np

# -----------------------------
# STEP 1: CREATE AIRPORT IMAGE
# -----------------------------
h, w = 600, 1000
bg = np.ones((h, w, 3), dtype=np.uint8) * 255

# Runway
cv2.rectangle(bg, (200, 250), (800, 350), (50, 50, 50), -1)

# Runway markings
for i in range(220, 780, 60):
    cv2.rectangle(bg, (i, 290), (i+30, 310), (255,255,255), -1)

# Taxiway
cv2.rectangle(bg, (200, 400), (800, 450), (80, 80, 80), -1)

cv2.imwrite("airport.png", bg)

# -----------------------------
# STEP 2: LOAD AIRCRAFT PNG
# -----------------------------
plane = cv2.imread("aeroplane.png", cv2.IMREAD_UNCHANGED)

if plane is None:
    print("❌ ERROR: aeroplane.png not found")
    exit()

plane = cv2.resize(plane, (80, 80))

# Handle different image formats
if len(plane.shape) == 2:
    print("❌ Grayscale image not supported")
    exit()

elif plane.shape[2] == 3:
    print("⚠ No alpha channel found. Creating one.")
    plane_rgb = plane
    plane_alpha = np.ones((plane.shape[0], plane.shape[1]))

elif plane.shape[2] == 4:
    plane_rgb = plane[:, :, :3]
    plane_alpha = plane[:, :, 3] / 255.0

# -----------------------------
# STEP 3: OVERLAY FUNCTION
# -----------------------------
def overlay(bg, fg, alpha, x, y):
    h, w = fg.shape[:2]

    # Boundary check
    if y + h > bg.shape[0] or x + w > bg.shape[1]:
        return bg

    for c in range(3):
        bg[y:y+h, x:x+w, c] = (
            alpha * fg[:, :, c] +
            (1 - alpha) * bg[y:y+h, x:x+w, c]
        )
    return bg

# -----------------------------
# STEP 4: VIDEO WRITER
# -----------------------------
out = cv2.VideoWriter(
    "simulation.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    20,
    (w, h)
)

# -----------------------------
# STEP 5: AIRCRAFT MOVEMENT
# -----------------------------
x, y = 50, 260   # starting position

for i in range(250):

    frame = bg.copy()

    # Draw aircraft
    frame = overlay(frame, plane_rgb, plane_alpha, int(x), int(y))

    # Movement logic
    if x < 600:
        x += 4  # runway movement
    else:
        y += 2  # turn to taxiway

    out.write(frame)

    # Optional: display
    cv2.imshow("Simulation", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

out.release()
cv2.destroyAllWindows()

print("✅ Simulation video saved as simulation.mp4")