import cv2
import json
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
media_dir = os.path.normpath(os.path.join(script_dir, "..", "Media"))

img_path = os.path.join(media_dir, "airport.jpg")
img = cv2.imread(img_path)
if img is None:
    print(f"Error: Image not found: {img_path}")
    exit()

orig_h, orig_w = img.shape[:2]

window_w, window_h = orig_w, orig_h
scale = 1.0

zoom = 1.0
view_x, view_y = 0, 0

mouse_x, mouse_y = 0, 0
dragging = False
is_drawing = False
start_x, start_y = 0, 0
last_x, last_y = 0, 0

restricted_zones = []
current_inner = None
current_outer = None
drawing_mode = "inner"
zone_counter = 1

colors = [
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (255, 255, 0),
    (0, 128, 128)
]

def get_color(idx):
    return colors[idx % len(colors)]

def get_box_coords(box):
    x = min(box[0], box[0] + box[2])
    y = min(box[1], box[1] + box[3])
    w = abs(box[2])
    h = abs(box[3])
    return x, y, w, h

def normalize_box(box):
    x1 = min(box[0], box[0] + box[2])
    y1 = min(box[1], box[1] + box[3])
    x2 = max(box[0], box[0] + box[2])
    y2 = max(box[1], box[1] + box[3])
    return x1, y1, x2, y2

def boxes_overlap(inner, outer):
    x1_i, y1_i, x2_i, y2_i = normalize_box(inner)
    x1_o, y1_o, x2_o, y2_o = normalize_box(outer)
    return not (x2_i < x1_o or x2_o < x1_i or y2_i < y1_o or y2_o < y1_i)

def mouse(event, x, y, flags, param):
    global zoom, view_x, view_y, dragging, last_x, last_y
    global mouse_x, mouse_y, is_drawing, start_x, start_y
    global current_inner, current_outer, drawing_mode, zone_counter, restricted_zones

    mouse_x, mouse_y = x, y

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom = min(zoom + 0.1, 3)
        else:
            zoom = max(zoom - 0.1, 0.5)

    elif event == cv2.EVENT_LBUTTONDOWN:
        real_x = int(view_x + x / zoom)
        real_y = int(view_y + y / zoom)
        start_x, start_y = real_x, real_y
        is_drawing = True
        dragging = True
        last_x, last_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
        real_x = int(view_x + x / zoom)
        real_y = int(view_y + y / zoom)
        if drawing_mode == "inner":
            current_inner = (start_x, start_y, real_x - start_x, real_y - start_y)
        else:
            current_outer = (start_x, start_y, real_x - start_x, real_y - start_y)

    elif event == cv2.EVENT_LBUTTONUP and is_drawing:
        is_drawing = False
        dragging = False
        if drawing_mode == "inner" and current_inner:
            x, y, w, h = get_box_coords(current_inner)
            if w > 10 and h > 10:
                drawing_mode = "outer"
                current_outer = None
                print(f"Inner zone drawn. Now draw OUTER warning zone...")
        elif drawing_mode == "outer" and current_outer:
            x, y, w, h = get_box_coords(current_outer)
            if w > 10 and h > 10:
                if current_inner and boxes_overlap(current_inner, current_outer):
                    xi, yi, wi, hi = get_box_coords(current_inner)
                    xo, yo, wo, ho = get_box_coords(current_outer)
                    
                    inner_area = wi * hi
                    outer_area = wo * ho
                    
                    if inner_area < outer_area:
                        zone_name = f"Zone {zone_counter}"
                        zone_data = {
                            "id": zone_counter,
                            "name": zone_name,
                            "restricted": {"x1": int(xi), "y1": int(yi), "x2": int(xi + wi), "y2": int(yi + hi)},
                            "warning": {"x1": int(xo), "y1": int(yo), "x2": int(xo + wo), "y2": int(yo + ho)}
                        }
                        restricted_zones.append(zone_data)
                        zone_counter += 1
                        print(f"Zone {len(restricted_zones)}: Restricted ({xi},{yi})-({xi+wi},{yi+hi}), Warning ({xo},{yo})-({xo+wo},{yo+ho})")
                        current_inner = None
                        current_outer = None
                        drawing_mode = "inner"
                    else:
                        print("Error: Inner zone must be smaller than outer zone! Redraw outer box.")
                        current_outer = None
                else:
                    print("Error: Boxes must overlap! Redraw outer box.")
                    current_outer = None
            else:
                current_outer = None

    elif event == cv2.EVENT_MOUSEMOVE and dragging and not is_drawing:
        dx = int((last_x - x) / zoom)
        dy = int((last_y - y) / zoom)
        view_x += dx
        view_y += dy
        last_x, last_y = x, y

    elif event == cv2.EVENT_RBUTTONDOWN:
        real_x = int(view_x + x / zoom)
        real_y = int(view_y + y / zoom)
        for i, zone in enumerate(restricted_zones):
            r = zone['restricted']
            if r['x1'] <= real_x <= r['x2'] and r['y1'] <= real_y <= r['y2']:
                print(f"Deleted: {zone['name']}")
                restricted_zones.pop(i)
                break

cv2.namedWindow("Restricted Zone Setup", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Restricted Zone Setup", window_w, window_h)
cv2.setMouseCallback("Restricted Zone Setup", mouse)

print("\n" + "=" * 60)
print("RESTRICTED ZONE SETUP (Dual Zone System)")
print("=" * 60)
print("Instructions:")
print("  1. Draw INNER box (restricted area)")
print("  2. Draw OUTER box (warning area)")
print("  3. Repeat for more zones")
print()
print("  LEFT CLICK & DRAG: Draw zone")
print("  RIGHT CLICK: Delete a zone")
print("  SCROLL: Zoom in/out")
print("  DRAG (no draw): Pan view")
print("  ENTER: Save and exit")
print("  ESC: Exit without saving")
print("  BACKSPACE: Undo last zone")
print("=" * 60)

while True:
    view_w = int(window_w / zoom)
    view_h = int(window_h / zoom)
    
    view_x_clamped = max(0, min(view_x, orig_w - view_w))
    view_y_clamped = max(0, min(view_y, orig_h - view_h))
    
    roi = img[view_y_clamped:view_y_clamped+view_h, view_x_clamped:view_x_clamped+view_w]
    display = cv2.resize(roi, (view_w, view_h))
    
    for i, zone in enumerate(restricted_zones):
        color = get_color(i)
        
        r = zone['restricted']
        ri = (int((r['x1'] - view_x_clamped) * zoom), int((r['y1'] - view_y_clamped) * zoom))
        rj = (int((r['x2'] - view_x_clamped) * zoom), int((r['y2'] - view_y_clamped) * zoom))
        overlay = display.copy()
        cv2.rectangle(overlay, ri, rj, (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
        cv2.rectangle(display, ri, rj, (0, 0, 255), 3)
        
        w = zone['warning']
        wi = (int((w['x1'] - view_x_clamped) * zoom), int((w['y1'] - view_y_clamped) * zoom))
        wj = (int((w['x2'] - view_x_clamped) * zoom), int((w['y2'] - view_y_clamped) * zoom))
        overlay = display.copy()
        cv2.rectangle(overlay, wi, wj, (0, 165, 255), -1)
        cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
        cv2.rectangle(display, wi, wj, (0, 165, 255), 2)
        
        cx = (ri[0] + rj[0]) // 2
        cy = ri[1] - 10
        cv2.putText(display, zone['name'], (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if current_inner:
        color = (0, 0, 255)
        x, y, w, h = get_box_coords(current_inner)
        ix = int((x - view_x_clamped) * zoom)
        iy = int((y - view_y_clamped) * zoom)
        iw = int(w * zoom)
        ih = int(h * zoom)
        cv2.rectangle(display, (ix, iy), (ix + iw, iy + ih), color, 2)
        cv2.putText(display, "INNER (Restricted)", (ix, iy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    if current_outer:
        color = (0, 165, 255)
        x, y, w, h = get_box_coords(current_outer)
        ix = int((x - view_x_clamped) * zoom)
        iy = int((y - view_y_clamped) * zoom)
        iw = int(w * zoom)
        ih = int(h * zoom)
        cv2.rectangle(display, (ix, iy), (ix + iw, iy + ih), color, 2)
        cv2.putText(display, "OUTER (Warning)", (ix, iy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    panel_h = 150
    panel = np.zeros((panel_h, 350, 3), dtype=np.uint8)
    cv2.addWeighted(panel, 0.6, panel, 0, 0, panel)
    display[0:panel_h, 0:350] = panel
    
    cv2.putText(display, "RESTRICTED ZONE SETUP", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(display, f"Zones: {len(restricted_zones)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display, f"Zoom: {zoom:.1f}x", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    mode_text = f"Mode: {drawing_mode.upper()}" if not (current_inner or current_outer) else "Draw box..."
    cv2.putText(display, mode_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(display, "ENTER: Save | ESC: Cancel", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(display, "BACKSPACE: Undo | R-Click: Delete", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    cv2.imshow("Restricted Zone Setup", display)
    key = cv2.waitKey(1)
    
    if key == 27:
        cv2.destroyAllWindows()
        exit()
    
    elif key == 13:
        if len(restricted_zones) > 0:
            break
        else:
            print("Please draw at least one restricted zone!")
    
    elif key == 8:
        if restricted_zones:
            removed = restricted_zones.pop()
            zone_counter -= 1
            print(f"Undo: Removed {removed['name']}")

cv2.destroyAllWindows()

save_path = os.path.join(script_dir, "restricted_zones.json")
with open(save_path, "w") as f:
    json.dump(restricted_zones, f, indent=2)

print("\n" + "=" * 60)
print("RESTRICTED ZONE SETUP COMPLETE")
print("=" * 60)
print(f"Saved {len(restricted_zones)} zone(s) to: {save_path}")
for zone in restricted_zones:
    r = zone['restricted']
    w = zone['warning']
    print(f"  {zone['name']}:")
    print(f"    Restricted: ({r['x1']},{r['y1']})-({r['x2']},{r['y2']})")
    print(f"    Warning: ({w['x1']},{w['y1']})-({w['x2']},{w['y2']})")
print("=" * 60)
