import cv2
import json
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
media_dir = os.path.normpath(os.path.join(script_dir, "..", "Media"))

img_path = os.path.join(media_dir, "parking.jpg")
img = cv2.imread(img_path)
if img is None:
    print(f"Error: Image not found: {img_path}")
    exit()

orig_h, orig_w = img.shape[:2]

window_w, window_h = 1200, 700
scale = min(window_w / orig_w, window_h / orig_h)

zoom = 1.0
view_x, view_y = 0, 0

mouse_x, mouse_y = 0, 0
dragging = False
is_drawing = False
start_x, start_y = 0, 0
last_x, last_y = 0, 0

terminal_boxes = []
current_box = None

colors = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 128),
    (255, 128, 0)
]

def get_color(idx):
    return colors[idx % len(colors)]

def mouse(event, x, y, flags, param):
    global zoom, view_x, view_y, dragging, last_x, last_y
    global mouse_x, mouse_y, is_drawing, start_x, start_y, current_box

    mouse_x, mouse_y = x, y

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom = min(zoom + 0.2, 5)
        else:
            zoom = max(zoom - 0.2, 1)

    elif event == cv2.EVENT_LBUTTONDOWN:
        real_x = int(view_x + x / (scale * zoom))
        real_y = int(view_y + y / (scale * zoom))
        start_x, start_y = real_x, real_y
        is_drawing = True
        dragging = True
        last_x, last_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
        real_x = int(view_x + x / (scale * zoom))
        real_y = int(view_y + y / (scale * zoom))
        current_box = (min(start_x, real_x), min(start_y, real_y),
                       abs(real_x - start_x), abs(real_y - start_y))

    elif event == cv2.EVENT_LBUTTONUP and is_drawing:
        is_drawing = False
        dragging = False
        if current_box and current_box[2] > 10 and current_box[3] > 10:
            terminal_boxes.append(current_box)
            print(f"Terminal {len(terminal_boxes)}: Box added at ({current_box[0]}, {current_box[1]}) size ({current_box[2]}x{current_box[3]})")
        current_box = None

    elif event == cv2.EVENT_MOUSEMOVE and dragging and not is_drawing:
        dx = int((last_x - x) / (scale * zoom))
        dy = int((last_y - y) / (scale * zoom))
        view_x += dx
        view_y += dy
        last_x, last_y = x, y

cv2.namedWindow("Terminal Setup")
cv2.setMouseCallback("Terminal Setup", mouse)

print("\n" + "=" * 50)
print("TERMINAL SETUP - Click & Drag to create boxes")
print("=" * 50)
print("Instructions:")
print("  - CLICK & DRAG: Draw terminal box")
print("  - SCROLL: Zoom in/out")
print("  - DRAG (no draw): Pan view")
print("  - ENTER: Save and exit")
print("  - ESC: Exit without saving")
print("=" * 50)

while True:
    view_w = int(window_w / (scale * zoom))
    view_h = int(window_h / (scale * zoom))
    
    view_x_clamped = max(0, min(view_x, orig_w - view_w))
    view_y_clamped = max(0, min(view_y, orig_h - view_h))
    
    roi = img[view_y_clamped:view_y_clamped+view_h, view_x_clamped:view_x_clamped+view_w]
    display = cv2.resize(roi, (window_w, window_h))
    
    for i, box in enumerate(terminal_boxes):
        color = get_color(i)
        bx = int((box[0] - view_x_clamped) * scale * zoom)
        by = int((box[1] - view_y_clamped) * scale * zoom)
        bw = int(box[2] * scale * zoom)
        bh = int(box[3] * scale * zoom)
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), color, 2)
        cv2.putText(display, f"T{i+1}", (bx + 5, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    if current_box:
        color = get_color(len(terminal_boxes))
        bx = int((current_box[0] - view_x_clamped) * scale * zoom)
        by = int((current_box[1] - view_y_clamped) * scale * zoom)
        bw = int(current_box[2] * scale * zoom)
        bh = int(current_box[3] * scale * zoom)
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), color, 2)
        cv2.putText(display, f"T{len(terminal_boxes)+1}", (bx + 5, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    info_h = 100
    cv2.rectangle(display, (0, 0), (260, info_h), (0, 0, 0), -1)
    cv2.putText(display, "TERMINAL SETUP", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, f"Terminals: {len(terminal_boxes)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display, f"Zoom: {zoom:.1f}x", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display, "Click & Drag to draw boxes", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    cv2.imshow("Terminal Setup", display)
    key = cv2.waitKey(1)
    
    if key == 27:
        cv2.destroyAllWindows()
        exit()
    
    elif key == 13:
        if len(terminal_boxes) > 0:
            break
        else:
            print("Please draw at least one terminal box!")

cv2.destroyAllWindows()

save_data = {
    "image_path": img_path,
    "image_width": orig_w,
    "image_height": orig_h,
    "terminal_boxes": terminal_boxes
}

save_path = os.path.join(script_dir, "parking.json")
with open(save_path, "w") as f:
    json.dump(save_data, f, indent=2)

print("\n" + "=" * 50)
print("TERMINAL SETUP COMPLETE")
print("=" * 50)
print(f"Saved {len(terminal_boxes)} terminal(s) to: {save_path}")
for i, box in enumerate(terminal_boxes):
    print(f"  Terminal {i+1}: ({box[0]}, {box[1]}) - {box[2]}x{box[3]}")
print("=" * 50)
