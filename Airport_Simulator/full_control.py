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

plane_path = os.path.join(media_dir, "aeroplane.png")
plane = cv2.imread(plane_path, cv2.IMREAD_UNCHANGED)
if plane is None:
    print(f"Error: Aircraft image not found: {plane_path}")
    exit()

window_w, window_h = orig_w, orig_h
scale = 1.0

zoom = 1.0
view_x, view_y = 0, 0

mouse_x, mouse_y = 0, 0
dragging = False
last_x, last_y = 0, 0

current_angle = 0
current_speed = 3
scale_plane = 0.1

aircraft_paths = []
aircraft_delays = []
aircraft_colors = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 128),
    (255, 128, 0),
    (0, 128, 255),
    (128, 255, 0)
]

def resize_plane(plane_img, scale_val):
    h, w = plane_img.shape[:2]
    new_w = max(10, int(w * scale_val))
    new_h = max(10, int(h * scale_val))
    return cv2.resize(plane_img, (new_w, new_h))

def rotate(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_TRANSPARENT)

def overlay(bg, fg, x, y):
    h, w = fg.shape[:2]
    if fg.shape[2] == 4:
        alpha = fg[:, :, 3] / 255.0
        fg_rgb = fg[:, :, :3]
    else:
        alpha = np.ones((h, w))
        fg_rgb = fg
    if y+h > bg.shape[0] or x+w > bg.shape[1] or x < 0 or y < 0:
        return bg
    for c in range(3):
        bg[y:y+h, x:x+w, c] = (alpha * fg_rgb[:, :, c] + (1 - alpha) * bg[y:y+h, x:x+w, c])
    return bg

def get_color(idx):
    return aircraft_colors[idx % len(aircraft_colors)]

current_path = []
current_aircraft_idx = 0
num_aircraft = 0

def mouse(event, x, y, flags, param):
    global zoom, view_x, view_y, dragging, last_x, last_y
    global mouse_x, mouse_y, current_path, current_aircraft_idx, num_aircraft
    global current_angle, current_speed

    mouse_x, mouse_y = x, y

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom = min(zoom + 0.1, 3)
        else:
            zoom = max(zoom - 0.1, 0.5)
        view_x = max(0, min(view_x, orig_w - int(window_w / zoom)))
        view_y = max(0, min(view_y, orig_h - int(window_h / zoom)))

    elif event == cv2.EVENT_LBUTTONDOWN:
        real_x = int(view_x + x / zoom)
        real_y = int(view_y + y / zoom)
        if 0 <= real_x < orig_w and 0 <= real_y < orig_h:
            current_path.append((real_x, real_y, current_angle, current_speed))
            color = get_color(current_aircraft_idx)
            print(f"Aircraft {current_aircraft_idx + 1}: Added point ({real_x}, {real_y}) angle={current_angle} speed={current_speed}")
        dragging = True
        last_x, last_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        dx = int((last_x - x) / zoom)
        dy = int((last_y - y) / zoom)
        view_x += dx
        view_y += dy
        view_x = max(0, min(view_x, orig_w - int(window_w / zoom)))
        view_y = max(0, min(view_y, orig_h - int(window_h / zoom)))
        last_x, last_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

def setup_aircraft_system():
    global num_aircraft, aircraft_delays
    
    print("\n" + "=" * 50)
    print("MULTIPLE AIRCRAFT SIMULATION SETUP")
    print("=" * 50)
    
    while True:
        try:
            num = int(input("Enter number of aircraft to simulate: "))
            if 1 <= num <= 10:
                num_aircraft = num
                break
            print("Please enter between 1 and 10")
        except ValueError:
            print("Please enter a valid number")
    
    print("\n" + "-" * 50)
    print("Enter TIME DELAYS for each aircraft (in seconds)")
    print("Aircraft will start at these delays from simulation start")
    print("-" * 50)
    
    aircraft_delays = []
    for i in range(num_aircraft):
        while True:
            try:
                delay = int(input(f"Aircraft {i + 1} delay (seconds): "))
                if delay >= 0:
                    aircraft_delays.append(delay)
                    break
                print("Delay must be 0 or positive")
            except ValueError:
                print("Please enter a valid number")
    
    print("\n" + "=" * 50)
    print("AIRRAFT PATH RECORDING")
    print("=" * 50)
    print("For each aircraft:")
    print("  - Click to add path points")
    print("  - Use A/D to change angle")
    print("  - Use W/S to change speed")
    print("  - Use Z/X to change size")
    print("  - Press ENTER when path is complete")
    print("  - ESC to exit")
    print("=" * 50)

cv2.namedWindow("Editor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Editor", window_w, window_h)
cv2.setMouseCallback("Editor", mouse)

setup_aircraft_system()

for i in range(num_aircraft):
    current_aircraft_idx = i
    current_path = []
    current_angle = 0
    current_speed = 3
    scale_plane = 1.0
    
    color = get_color(i)
    print(f"\n>>> RECORDING PATH FOR AIRCRAFT {i + 1} (delay: {aircraft_delays[i]}s) <<<")
    
    while True:
        view_w = int(window_w / zoom)
        view_h = int(window_h / zoom)
        view_x = max(0, min(view_x, orig_w - view_w))
        view_y = max(0, min(view_y, orig_h - view_h))
        roi = img[view_y:view_y+view_h, view_x:view_x+view_w]
        display = cv2.resize(roi, (window_w, window_h))
        
        for j, path in enumerate(aircraft_paths):
            path_color = get_color(j)
            for k in range(1, len(path)):
                p1, p2 = path[k-1], path[k]
                x1 = int((p1[0] - view_x) * zoom)
                y1 = int((p1[1] - view_y) * zoom)
                x2 = int((p2[0] - view_x) * zoom)
                y2 = int((p2[1] - view_y) * zoom)
                cv2.line(display, (x1, y1), (x2, y2), path_color, 2)
                if k == 1 or k == len(path) - 1:
                    cv2.circle(display, (x1, y1), 5, path_color, -1)
        
        for k in range(1, len(current_path)):
            p1, p2 = current_path[k-1], current_path[k]
            x1 = int((p1[0] - view_x) * zoom)
            y1 = int((p1[1] - view_y) * zoom)
            x2 = int((p2[0] - view_x) * zoom)
            y2 = int((p2[1] - view_y) * zoom)
            cv2.line(display, (x1, y1), (x2, y2), color, 2)
            cv2.circle(display, (x1, y1), 5, color, -1)
        
        real_x = int(view_x + mouse_x / zoom)
        real_y = int(view_y + mouse_y / zoom)
        preview = resize_plane(plane, scale_plane)
        preview = rotate(preview, current_angle)
        px = int((real_x - view_x) * zoom)
        py = int((real_y - view_y) * zoom)
        display = overlay(display, preview, px, py)
        
        panel_h = 160
        cv2.rectangle(display, (0, 0), (280, panel_h), (0, 0, 0), -1)
        cv2.putText(display, f"Aircraft: {i + 1}/{num_aircraft}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display, f"Delay: {aircraft_delays[i]}s | Zoom: {zoom:.1f}x", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"Angle: {current_angle} | A/D: adjust", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"Speed: {current_speed} | W/S: adjust", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"Size: {round(scale_plane,2)} | Z/X: adjust", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"Points: {len(current_path)} | SCROLL: zoom", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Editor", display)
        key = cv2.waitKey(1)
        
        if key == 27:
            cv2.destroyAllWindows()
            exit()
        
        elif key == ord('a'):
            current_angle -= 10
        elif key == ord('d'):
            current_angle += 10
        elif key == ord('w'):
            current_speed += 1
        elif key == ord('s'):
            current_speed = max(0.1, current_speed - 0.5)
        elif key == ord('z'):
            scale_plane += 0.1
        elif key == ord('x'):
            scale_plane = max(0.1, scale_plane - 0.1)
        
        elif key == 13:
            if len(current_path) >= 2:
                aircraft_paths.append(current_path.copy())
                print(f"Aircraft {i + 1}: Path recorded with {len(current_path)} points")
                break
            else:
                print("Need at least 2 points for a path!")
    
    view_x, view_y = 0, 0
    zoom = 1.0

cv2.destroyAllWindows()

save_data = {
    "aircraft_paths": aircraft_paths,
    "aircraft_delays": aircraft_delays,
    "scale_plane": scale_plane,
    "current_speed": current_speed
}

with open("simulation_path.json", "w") as f:
    json.dump(save_data, f, indent=2)

print("\nSaved simulation route to simulation_path.json")

out = cv2.VideoWriter("simulation.mp4", 0x00000021, 30, (orig_w, orig_h))

cv2.namedWindow("Simulation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Simulation", window_w, window_h)

print("\n" + "=" * 50)
print("RUNNING SIMULATION")
print("=" * 50)
print(f"Aircraft delays: {aircraft_delays}")
for i, path in enumerate(aircraft_paths):
    print(f"Aircraft {i + 1}: {len(path)} points")
print("-" * 50)

frame_rate = 30
max_delay = max(aircraft_delays) if aircraft_delays else 0
total_frames = 0

for i, path in enumerate(aircraft_paths):
    if len(path) >= 2:
        path_frames = 0
        for j in range(len(path) - 1):
            p1, p2 = path[j], path[j + 1]
            dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            path_frames += max(1, int(dist / p1[3]))
        total_frames = max(total_frames, aircraft_delays[i] * frame_rate + path_frames)

aircraft_progress = [0.0] * num_aircraft
aircraft_path_idx = [0] * num_aircraft
aircraft_started = [False] * num_aircraft
aircraft_completed = [False] * num_aircraft
aircraft_final_pos = [(0, 0, 0)] * num_aircraft
aircraft_current_angle = [0.0] * num_aircraft

for frame_num in range(total_frames):
    frame = img.copy()
    
    for i in range(num_aircraft):
        if aircraft_completed[i]:
            continue
        
        if not aircraft_started[i]:
            if frame_num >= aircraft_delays[i] * frame_rate:
                aircraft_started[i] = True
                print(f"Aircraft {i + 1} started at frame {frame_num}")
            else:
                continue
        
        if aircraft_path_idx[i] >= len(aircraft_paths[i]) - 1:
            aircraft_completed[i] = True
            aircraft_final_pos[i] = (aircraft_paths[i][-1][0], aircraft_paths[i][-1][1], aircraft_paths[i][-1][2])
            aircraft_current_angle[i] = aircraft_paths[i][-1][2]
            print(f"Aircraft {i + 1} completed")
            continue
        
        p1 = aircraft_paths[i][aircraft_path_idx[i]]
        p2 = aircraft_paths[i][aircraft_path_idx[i] + 1]
        
        dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        steps = max(1, int(dist / p1[3]))
        
        aircraft_progress[i] += 1.0 / steps
        
        if aircraft_progress[i] >= 1.0:
            aircraft_progress[i] = 0.0
            aircraft_path_idx[i] += 1
            if aircraft_path_idx[i] >= len(aircraft_paths[i]) - 1:
                aircraft_completed[i] = True
                aircraft_final_pos[i] = (aircraft_paths[i][-1][0], aircraft_paths[i][-1][1], aircraft_paths[i][-1][2])
                aircraft_current_angle[i] = aircraft_paths[i][-1][2]
                print(f"Aircraft {i + 1} completed")
            continue
        
        t = aircraft_progress[i]
        x = int((1 - t) * p1[0] + t * p2[0])
        y = int((1 - t) * p1[1] + t * p2[1])
        
        target_angle = p2[2]
        current_angle_val = aircraft_current_angle[i]
        diff = target_angle - current_angle_val
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        smooth_step = 5
        aircraft_current_angle[i] += diff / smooth_step
        while aircraft_current_angle[i] > 180:
            aircraft_current_angle[i] -= 360
        while aircraft_current_angle[i] < -180:
            aircraft_current_angle[i] += 360
        
        plane_scaled = resize_plane(plane, scale_plane)
        rotated = rotate(plane_scaled, aircraft_current_angle[i])
        frame = overlay(frame, rotated, x, y)
    
    for i in range(num_aircraft):
        if aircraft_completed[i] and aircraft_started[i]:
            x, y, angle = aircraft_final_pos[i]
            plane_scaled = resize_plane(plane, scale_plane)
            rotated = rotate(plane_scaled, angle)
            frame = overlay(frame, rotated, x, y)
    
    frame_out = frame.copy()
    
    for i in range(num_aircraft):
        if not aircraft_started[i]:
            continue
        
        if aircraft_completed[i]:
            x, y, _ = aircraft_final_pos[i]
        elif aircraft_path_idx[i] < len(aircraft_paths[i]) - 1:
            p1 = aircraft_paths[i][aircraft_path_idx[i]]
            p2 = aircraft_paths[i][aircraft_path_idx[i] + 1]
            t = aircraft_progress[i]
            x = int((1 - t) * p1[0] + t * p2[0])
            y = int((1 - t) * p1[1] + t * p2[1])
        else:
            continue
        
        color = get_color(i)
        cv2.putText(frame_out, f"A{i + 1}", (x - 20, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imshow("Simulation", frame_out)
    out.write(frame_out)
    
    if cv2.waitKey(1) == 27:
        break

out.release()
cv2.destroyAllWindows()

print("-" * 50)
print("Simulation complete")
print("Simulation video saved as simulation.mp4")
