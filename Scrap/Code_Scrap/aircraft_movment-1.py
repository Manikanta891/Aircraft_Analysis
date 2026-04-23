# This code is correct and functioning properly and it has the olden code for the aircraft movement simulation. 
# It creates a simple simulation of an aircraft moving along a user-defined path on an airport image. 
# The user can zoom and pan the image to select the start and end points for the aircraft's movement. 
# The simulation then animates the aircraft moving along the defined path, and it saves the animation as a video file. 
# This code is not used in the final Streamlit app but serves as a demonstration of how to create a basic aircraft movement simulation using OpenCV.

# import cv2
# import numpy as np

# # -----------------------------
# # LOAD AIRPORT IMAGE
# # -----------------------------
# img = cv2.imread("airport.jpg")

# if img is None:
#     print("❌ Airport image not found")
#     exit()

# clone = img.copy()

# # -----------------------------
# # LOAD AIRCRAFT PNG
# # -----------------------------
# plane = cv2.imread("aeroplane.png", cv2.IMREAD_UNCHANGED)

# if plane is None:
#     print("❌ Aircraft image not found")
#     exit()

# plane = cv2.resize(plane, (60, 60))

# if plane.shape[2] == 4:
#     plane_rgb = plane[:, :, :3]
#     plane_alpha = plane[:, :, 3] / 255.0
# else:
#     plane_rgb = plane
#     plane_alpha = np.ones((plane.shape[0], plane.shape[1]))

# # -----------------------------
# # ROTATE FUNCTION
# # -----------------------------
# def rotate_image(image, angle):
#     h, w = image.shape[:2]
#     center = (w // 2, h // 2)

#     matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(
#         image, matrix, (w, h),
#         flags=cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_TRANSPARENT
#     )
#     return rotated

# # -----------------------------
# # OVERLAY FUNCTION
# # -----------------------------
# def overlay(bg, fg, alpha, x, y):
#     h, w = fg.shape[:2]

#     if y + h > bg.shape[0] or x + w > bg.shape[1]:
#         return bg

#     for c in range(3):
#         bg[y:y+h, x:x+w, c] = (
#             alpha * fg[:, :, c] +
#             (1 - alpha) * bg[y:y+h, x:x+w, c]
#         )
#     return bg

# # -----------------------------
# # MOUSE INPUT
# # -----------------------------
# points = []

# def click_event(event, x, y, flags, param):
#     global points

#     if event == cv2.EVENT_LBUTTONDOWN:
#         if len(points) < 2:
#             points.append((x, y))
#             print(f"Selected: {x}, {y}")

# cv2.namedWindow("Select Path")
# cv2.setMouseCallback("Select Path", click_event)

# print("👉 Click START and END points")

# # Select points
# while True:
#     temp = img.copy()

#     for p in points:
#         cv2.circle(temp, p, 6, (0,0,255), -1)

#     if len(points) == 2:
#         cv2.line(temp, points[0], points[1], (255,0,0), 2)

#     cv2.imshow("Select Path", temp)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

#     if len(points) == 2:
#         break

# cv2.destroyAllWindows()

# # -----------------------------
# # SIMULATION
# # -----------------------------
# start = np.array(points[0])
# end = np.array(points[1])

# dx = end[0] - start[0]
# dy = end[1] - start[1]

# angle = np.degrees(np.arctan2(-dy, dx))

# steps = 200

# for i in range(steps):
#     frame = clone.copy()

#     t = i / steps
#     pos = (1 - t) * start + t * end
#     x, y = int(pos[0]), int(pos[1])

#     rotated_plane = rotate_image(plane_rgb, angle)

#     frame = overlay(frame, rotated_plane, plane_alpha, x, y)

#     cv2.line(frame, tuple(start), tuple(end), (255,0,0), 2)

#     cv2.imshow("Aircraft Simulation", frame)

#     if cv2.waitKey(30) & 0xFF == 27:
#         break

# cv2.destroyAllWindows()

# print("✅ Simulation completed")

import cv2
import numpy as np

# -----------------------------
# LOAD IMAGE
# -----------------------------
img = cv2.imread("airport.jpg")

if img is None:
    print("❌ Image not found")
    exit()

orig_h, orig_w = img.shape[:2]

# -----------------------------
# WINDOW SIZE (screen fit)
# -----------------------------
window_w, window_h = 1200, 700

# Initial scale to fit full image
scale = min(window_w / orig_w, window_h / orig_h)

# Zoom variables
zoom = 1.0
min_zoom, max_zoom = 1.0, 5.0

view_x, view_y = 0, 0

points = []

dragging = False
last_x, last_y = 0, 0

# -----------------------------
# MOUSE FUNCTION
# -----------------------------
def mouse(event, x, y, flags, param):
    global zoom, view_x, view_y, dragging, last_x, last_y, points

    if event == cv2.EVENT_MOUSEWHEEL:
        old_zoom = zoom

        if flags > 0:
            zoom = min(zoom + 0.2, max_zoom)
        else:
            zoom = max(zoom - 0.2, min_zoom)

        # Zoom towards cursor
        mx, my = x, y

        view_x = int(view_x + (mx / (scale * old_zoom)) - (mx / (scale * zoom)))
        view_y = int(view_y + (my / (scale * old_zoom)) - (my / (scale * zoom)))

        # Clamp
        max_x = orig_w - int(window_w / (scale * zoom))
        max_y = orig_h - int(window_h / (scale * zoom))

        view_x = max(0, min(view_x, max_x))
        view_y = max(0, min(view_y, max_y))

    elif event == cv2.EVENT_LBUTTONDOWN:
        # Convert to original coordinates
        real_x = int(view_x + x / (scale * zoom))
        real_y = int(view_y + y / (scale * zoom))

        if len(points) < 2:
            points.append((real_x, real_y))
            print(f"Selected: {real_x}, {real_y}")

        dragging = True
        last_x, last_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            dx = int((last_x - x) / (scale * zoom))
            dy = int((last_y - y) / (scale * zoom))

            view_x = max(0, min(view_x + dx, orig_w - int(window_w / (scale * zoom))))
            view_y = max(0, min(view_y + dy, orig_h - int(window_h / (scale * zoom))))

            last_x, last_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False


# -----------------------------
# INTERACTION WINDOW
# -----------------------------
cv2.namedWindow("Select Path")
cv2.setMouseCallback("Select Path", mouse)

print("👉 Scroll to zoom | Drag to move | Click 2 points | Press ENTER")

while True:

    view_w = int(window_w / (scale * zoom))
    view_h = int(window_h / (scale * zoom))

    roi = img[view_y:view_y+view_h, view_x:view_x+view_w]
    display = cv2.resize(roi, (window_w, window_h))

    # Draw selected points
    for p in points:
        px = int((p[0] - view_x) * scale * zoom)
        py = int((p[1] - view_y) * scale * zoom)
        cv2.circle(display, (px, py), 6, (0,0,255), -1)

    if len(points) == 2:
        p1, p2 = points
        px1 = int((p1[0] - view_x) * scale * zoom)
        py1 = int((p1[1] - view_y) * scale * zoom)
        px2 = int((p2[0] - view_x) * scale * zoom)
        py2 = int((p2[1] - view_y) * scale * zoom)


    cv2.putText(display, "Zoom: Scroll | Move: Drag | Click 2 points | ENTER",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Select Path", display)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        exit()
    if key == 13 and len(points) == 2:  # ENTER
        break

cv2.destroyAllWindows()

# -----------------------------
# LOAD AIRCRAFT
# -----------------------------
plane = cv2.imread("aeroplane.png", cv2.IMREAD_UNCHANGED)
plane = cv2.resize(plane, (60, 60))

if plane.shape[2] == 4:
    plane_rgb = plane[:, :, :3]
    plane_alpha = plane[:, :, 3] / 255.0
else:
    plane_rgb = plane
    plane_alpha = np.ones((plane.shape[0], plane.shape[1]))

# -----------------------------
# HELPERS
# -----------------------------
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h),
                          borderMode=cv2.BORDER_TRANSPARENT)

def overlay(bg, fg, alpha, x, y):
    h, w = fg.shape[:2]
    if y+h > bg.shape[0] or x+w > bg.shape[1]:
        return bg

    for c in range(3):
        bg[y:y+h, x:x+w, c] = (
            alpha * fg[:, :, c] +
            (1 - alpha) * bg[y:y+h, x:x+w, c]
        )
    return bg

# -----------------------------
# SIMULATION (FULL IMAGE VIEW)
# -----------------------------
start = np.array(points[0])
end = np.array(points[1])

dx = end[0] - start[0]
dy = end[1] - start[1]
angle = np.degrees(np.arctan2(-dy, dx))

steps = 200
# -----------------------------
# VIDEO WRITER
# -----------------------------
fps = 30
out = cv2.VideoWriter(
    "aircraft_simulation.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (window_w, window_h)
)

# -----------------------------
# SIMULATION
# -----------------------------
steps = 200

for i in range(steps):
    frame = img.copy()

    t = i / steps
    pos = (1 - t) * start + t * end
    x, y = int(pos[0]), int(pos[1])

    rotated = rotate_image(plane_rgb, angle)
    frame = overlay(frame, rotated, plane_alpha, x, y)

    cv2.line(frame, tuple(start), tuple(end), (255,0,0), 2)

    display = cv2.resize(frame, (window_w, window_h))

    cv2.imshow("Simulation", display)

    # SAVE VIDEO
    out.write(display)

    if cv2.waitKey(30) & 0xFF == 27:
        break

out.release()
cv2.destroyAllWindows()

print("✅ Video saved as aircraft_simulation.mp4")