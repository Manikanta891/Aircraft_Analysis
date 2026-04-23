# This code is a complete representation of an airport layout, including runways, taxiways, terminal building, gates, parking area, and control tower. 
# It uses OpenCV to draw the various components of the airport on a canvas and saves the resulting image as "airport_full.png".

import cv2
import numpy as np

# Canvas
h, w = 700, 1200
img = np.ones((h, w, 3), dtype=np.uint8) * 255

# -----------------------------
# RUNWAY (Main)
# -----------------------------
cv2.rectangle(img, (100, 250), (1100, 350), (50, 50, 50), -1)

# Runway center dashed line
for i in range(120, 1080, 80):
    cv2.rectangle(img, (i, 295), (i+40, 305), (255,255,255), -1)

# Runway numbers
cv2.putText(img, "09", (110, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
cv2.putText(img, "27", (1020, 330), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# -----------------------------
# SECOND RUNWAY (Cross)
# -----------------------------
cv2.rectangle(img, (500, 100), (600, 600), (60, 60, 60), -1)

# Center line
for i in range(120, 580, 60):
    cv2.rectangle(img, (545, i), (555, i+30), (255,255,255), -1)

# -----------------------------
# TAXIWAYS
# -----------------------------
# Horizontal taxiway
cv2.rectangle(img, (100, 400), (1100, 450), (80, 80, 80), -1)

# Vertical taxiway
cv2.rectangle(img, (300, 100), (350, 600), (90, 90, 90), -1)

# Taxiway lines (yellow)
for i in range(120, 1080, 100):
    cv2.line(img, (i, 425), (i+50, 425), (0,255,255), 2)

# -----------------------------
# TERMINAL BUILDING
# -----------------------------
cv2.rectangle(img, (50, 500), (400, 680), (200, 200, 200), -1)
cv2.putText(img, "TERMINAL", (120, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

# -----------------------------
# GATES
# -----------------------------
gate_x = 80
for i in range(5):
    cv2.rectangle(img, (gate_x, 470), (gate_x+40, 500), (150,150,150), -1)
    cv2.putText(img, f"G{i+1}", (gate_x, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    gate_x += 70

# -----------------------------
# PARKING AREA
# -----------------------------
cv2.rectangle(img, (450, 500), (1100, 680), (220, 220, 220), -1)
cv2.putText(img, "PARKING", (700, 600),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

# Parking slots
for i in range(480, 1080, 80):
    cv2.rectangle(img, (i, 520), (i+40, 650), (255,255,255), 2)

# -----------------------------
# CONTROL TOWER
# -----------------------------
cv2.circle(img, (1000, 100), 40, (180,180,180), -1)
cv2.putText(img, "TWR", (970, 105),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

# -----------------------------
# LABEL AIRPORT
# -----------------------------
cv2.putText(img, "AIRPORT - A", (450, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

# -----------------------------
# SAVE & SHOW
# -----------------------------
cv2.imshow("Airport Top View", img)
cv2.imwrite("airport_full.png", img)

cv2.waitKey(0)
cv2.destroyAllWindows()