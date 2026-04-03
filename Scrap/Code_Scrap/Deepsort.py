# import cv2
# import numpy as np
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort

# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# model = YOLO("C:\\Users\\manik\\Desktop\\Final Year Project\\Aeroplane\\Aircraft_Project-2\\aircraft_analysis_mani\\aircraft_detector.pt")   # <-- your YOLO model
# tracker = DeepSort(max_age=30)

# # -----------------------------
# # VIDEO INPUT
# # -----------------------------
# cap = cv2.VideoCapture("aircraft_simulation.mp4")

# # Sliding window settings
# tile_size = 640
# step = 500   # overlap (important)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     H, W, _ = frame.shape
#     detections = []

#     # -----------------------------
#     # SLIDING WINDOW
#     # -----------------------------
#     for y in range(0, H, step):
#         for x in range(0, W, step):

#             patch = frame[y:y+tile_size, x:x+tile_size]

#             # Skip small patches
#             if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
#                 continue

#             results = model(patch, conf=0.25)[0]

#             if results.boxes is None:
#                 continue

#             for r in results.boxes.data:
#                 x1, y1, x2, y2, conf, cls = r.tolist()

#                 # Convert to global coordinates
#                 x1 += x
#                 y1 += y
#                 x2 += x
#                 y2 += y

#                 w = x2 - x1
#                 h = y2 - y1

#                 detections.append(([x1, y1, w, h], conf, 'aircraft'))

#     # -----------------------------
#     # DEEPSORT TRACKING
#     # -----------------------------
#     tracks = tracker.update_tracks(detections, frame=frame)

#     for track in tracks:
#         if not track.is_confirmed():
#             continue

#         track_id = track.track_id
#         l, t, w, h = track.to_ltrb()

#         # Draw box
#         cv2.rectangle(frame, (int(l), int(t)),
#                       (int(l+w), int(t+h)), (0,255,0), 2)

#         # ID
#         cv2.putText(frame, f"ID: {track_id}",
#                     (int(l), int(t)-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

#     # -----------------------------
#     # DISPLAY
#     # -----------------------------
#     display = cv2.resize(frame, (1200, 700))
#     cv2.imshow("Sliding Window Tracking", display)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort

# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# model = YOLO("aircraft_detector.pt")   # <-- replace with your model
# tracker = DeepSort(max_age=30)

# # -----------------------------
# # VIDEO INPUT
# # -----------------------------
# cap = cv2.VideoCapture("fc_simulation-2.mp4")

# # -----------------------------
# # SLIDING WINDOW SETTINGS
# # -----------------------------
# tile_size = 640
# step = 500   # overlap

# # -----------------------------
# # ANALYSIS STORAGE
# # -----------------------------
# track_history = {}
# moving_ids = set()
# stationary_ids = set()

# stationary_threshold = 10  # pixels

# # -----------------------------
# # MAIN LOOP
# # -----------------------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     H, W, _ = frame.shape
#     detections = []

#     # -----------------------------
#     # SLIDING WINDOW DETECTION
#     # -----------------------------
#     for y in range(0, H, step):
#         for x in range(0, W, step):

#             patch = frame[y:y+tile_size, x:x+tile_size]

#             if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
#                 continue

#             results = model(patch, conf=0.25)[0]

#             if results.boxes is None:
#                 continue

#             for r in results.boxes.data:
#                 x1, y1, x2, y2, conf, cls = r.tolist()

#                 # Convert to global coordinates
#                 x1 += x
#                 y1 += y
#                 x2 += x
#                 y2 += y

#                 w = x2 - x1
#                 h = y2 - y1

#                 detections.append(([x1, y1, w, h], conf, 'aircraft'))

#     # -----------------------------
#     # TRACKING
#     # -----------------------------
#     tracks = tracker.update_tracks(detections, frame=frame)

#     for track in tracks:
#         if not track.is_confirmed():
#             continue

#         track_id = track.track_id
#         l, t, w, h = track.to_ltrb()

#         cx = int(l + w / 2)
#         cy = int(t + h / 2)

#         # -----------------------------
#         # STORE HISTORY
#         # -----------------------------
#         if track_id not in track_history:
#             track_history[track_id] = []

#         track_history[track_id].append((cx, cy))

#         # keep last 10 points
#         if len(track_history[track_id]) > 10:
#             track_history[track_id].pop(0)

#         # -----------------------------
#         # MOVEMENT ANALYSIS
#         # -----------------------------
#         if len(track_history[track_id]) >= 2:
#             x1, y1 = track_history[track_id][0]
#             x2, y2 = track_history[track_id][-1]

#             dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

#             if dist < stationary_threshold:
#                 stationary_ids.add(track_id)
#             else:
#                 moving_ids.add(track_id)

#         # -----------------------------
#         # DRAW BOUNDING BOX
#         # -----------------------------
#         cv2.rectangle(frame, (int(l), int(t)),
#                       (int(l+w), int(t+h)), (0,255,0), 2)

#         cv2.putText(frame, f"ID: {track_id}",
#                     (int(l), int(t)-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

#     # -----------------------------
#     # DRAW TRAJECTORY
#     # -----------------------------
#     for tid, pts in track_history.items():
#         for i in range(1, len(pts)):
#             cv2.line(frame, pts[i-1], pts[i], (255,0,0), 2)

#     # -----------------------------
#     # ANALYSIS DISPLAY
#     # -----------------------------
#     total_aircraft = len(track_history)

#     cv2.putText(frame, f"Total Aircraft: {total_aircraft}",
#                 (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

#     cv2.putText(frame, f"Moving: {len(moving_ids)}",
#                 (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

#     cv2.putText(frame, f"Stationary: {len(stationary_ids)}",
#                 (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

#     # -----------------------------
#     # DISPLAY
#     # -----------------------------
#     display = cv2.resize(frame, (1200, 700))
#     cv2.imshow("Aircraft Analysis System", display)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort

# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# model = YOLO("aircraft_detector.pt")
# tracker = DeepSort(max_age=10, n_init=3)

# cap = cv2.VideoCapture("fc_simulation-2.mp4")

# tile_size = 640
# step = 300

# # -----------------------------
# # NMS FUNCTION (REMOVE DUPLICATES)
# # -----------------------------
# def nms(detections, iou_thresh=0.6):
#     if len(detections) == 0:
#         return []

#     boxes = []
#     scores = []
    


#     for det in detections:
#         (x, y, w, h), conf, _ = det
#         boxes.append([x, y, x+w, y+h])
#         scores.append(conf)
        
#     for det in detections:
#         (x, y, w, h), conf, _ = det
#         cv2.rectangle(frame, (int(x), int(y)),
#                 (int(x+w), int(y+h)),
#                 (255,0,0), 1)

#     boxes = np.array(boxes)
#     scores = np.array(scores)

#     indices = cv2.dnn.NMSBoxes(
#         bboxes=boxes.tolist(),
#         scores=scores.tolist(),
#         score_threshold=0.2,
#         nms_threshold=iou_thresh
#     )

#     final = []
#     if len(indices) > 0:
#         for i in indices.flatten():
#             final.append(detections[i])

#     return final

# # -----------------------------
# # MAIN LOOP
# # -----------------------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     H, W, _ = frame.shape
#     detections = []

#     # -----------------------------
#     # SLIDING WINDOW DETECTION
#     # -----------------------------
#     for y in range(0, H, step):
#         for x in range(0, W, step):

#             patch = frame[y:y+tile_size, x:x+tile_size]

#             if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
#                 continue

#             results = model(patch, conf=0.1, imgsz=960)[0]
#             if results.boxes is None:
#                 continue

#             for r in results.boxes.data:
#                 x1, y1, x2, y2, conf, cls = r.tolist()

#                 # convert to global
#                 x1 += x
#                 y1 += y
#                 x2 += x
#                 y2 += y

#                 w = x2 - x1
#                 h = y2 - y1

#                 detections.append(([x1, y1, w, h], conf, 'aircraft'))

#     # -----------------------------
#     # REMOVE DUPLICATES (CRITICAL FIX)
#     # -----------------------------
#     detections = nms(detections)

#     # -----------------------------
#     # TRACKING
#     # -----------------------------
#     tracks = tracker.update_tracks(detections, frame=frame)

#     current_ids = set()

#     for track in tracks:
#         if not track.is_confirmed():
#             continue

#         # filter weak tracks
#         if track.hits < 3:
#             continue

#         track_id = track.track_id
#         current_ids.add(track_id)

#         l, t, w, h = track.to_ltrb()

#         # 🔥 shrink box slightly (fix large box issue)
#         shrink = 0.1
#         l = l + w*shrink
#         t = t + h*shrink
#         w = w * (1 - 2*shrink)
#         h = h * (1 - 2*shrink)

#         cv2.rectangle(frame,
#                       (int(l), int(t)),
#                       (int(l+w), int(t+h)),
#                       (0,255,0), 2)

#     # -----------------------------
#     # COUNT ONLY CURRENT AIRCRAFT
#     # -----------------------------
#     total_now = len(current_ids)

#     cv2.putText(frame, f"Aircraft Now: {total_now}",
#                 (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

#     # -----------------------------
#     # DISPLAY
#     # -----------------------------
#     display = cv2.resize(frame, (1200, 700))
#     cv2.imshow("Aircraft Detection (Clean)", display)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort

# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# model = YOLO("aircraft_detector.pt")

# tracker = DeepSort(
#     max_age=10,
#     n_init=3
# )

# # -----------------------------
# # VIDEO INPUT
# # -----------------------------
# cap = cv2.VideoCapture("fc_simulation-2.mp4")

# # -----------------------------
# # MAIN LOOP
# # -----------------------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # -----------------------------
#     # YOLO DETECTION (FULL FRAME)
#     # -----------------------------
#     results = model(frame, conf=0.15, imgsz=960)[0]

#     detections = []

#     if results.boxes is not None:
#         for r in results.boxes.data:
#             x1, y1, x2, y2, conf, cls = r.tolist()

#             w = x2 - x1
#             h = y2 - y1

#             detections.append(([x1, y1, w, h], conf, 'aircraft'))

#     # -----------------------------
#     # TRACKING
#     # -----------------------------
#     tracks = tracker.update_tracks(detections, frame=frame)

#     current_positions = []
#     threshold = 50   # distance threshold

#     # -----------------------------
#     # PROCESS TRACKS
#     # -----------------------------
#     for track in tracks:

#         if not track.is_confirmed():
#             continue

#         if track.hits < 3:
#             continue

#         l, t, w, h = track.to_ltrb()

#         cx = int(l + w/2)
#         cy = int(t + h/2)

#         # -----------------------------
#         # SPATIAL COUNTING (IMPORTANT)
#         # -----------------------------
#         is_new = True
#         for px, py in current_positions:
#             dist = np.sqrt((cx - px)**2 + (cy - py)**2)
#             if dist < threshold:
#                 is_new = False
#                 break

#         if is_new:
#             current_positions.append((cx, cy))

#         # -----------------------------
#         # DRAW BOX (SMALLER BOX FIX)
#         # -----------------------------
#         shrink = 0.1

#         l = l + w * shrink
#         t = t + h * shrink
#         w = w * (1 - 2 * shrink)
#         h = h * (1 - 2 * shrink)

#         cv2.rectangle(frame,
#                       (int(l), int(t)),
#                       (int(l+w), int(t+h)),
#                       (0,255,0), 2)

#     # -----------------------------
#     # CURRENT COUNT ONLY
#     # -----------------------------
#     total_now = len(current_positions)

#     cv2.putText(frame, f"Aircraft Now: {total_now}",
#                 (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8, (0,0,255), 2)

#     # -----------------------------
#     # DISPLAY
#     # -----------------------------
#     display = cv2.resize(frame, (1200, 700))
#     cv2.imshow("Aircraft Detection (Clean)", display)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()


# This code snippet is correct the above one 

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort

# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# model = YOLO("aircraft_detector.pt")

# tracker = DeepSort(
#     max_age=10,
#     n_init=3
# )

# # -----------------------------
# # VIDEO INPUT
# # -----------------------------
# cap = cv2.VideoCapture("fc_simulation-2.mp4")

# # -----------------------------
# # MAIN LOOP
# # -----------------------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # -----------------------------
#     # YOLO DETECTION (FULL FRAME)
#     # -----------------------------
#     results = model(frame, conf=0.15, imgsz=960)[0]

#     detections = []
#     yolo_boxes = []

#     if results.boxes is not None:
#         for r in results.boxes.data:
#             x1, y1, x2, y2, conf, cls = r.tolist()

#             w = x2 - x1
#             h = y2 - y1

#             # DeepSORT format
#             detections.append(([x1, y1, w, h], conf, 'aircraft'))

#             # Store YOLO box + center
#             cx = int((x1 + x2) / 2)
#             cy = int((y1 + y2) / 2)
#             yolo_boxes.append((x1, y1, x2, y2, cx, cy))

#     # -----------------------------
#     # TRACKING
#     # -----------------------------
#     tracks = tracker.update_tracks(detections, frame=frame)

#     current_positions = []
#     threshold = 50  # distance threshold

#     # -----------------------------
#     # MATCH TRACKS WITH YOLO BOXES
#     # -----------------------------
#     for track in tracks:

#         if not track.is_confirmed():
#             continue

#         if track.hits < 3:
#             continue

#         track_id = track.track_id

#         # Tracker center
#         l, t, w, h = track.to_ltrb()
#         tx = int(l + w / 2)
#         ty = int(t + h / 2)

#         # Find nearest YOLO box
#         best_box = None
#         min_dist = float('inf')

#         for box in yolo_boxes:
#             x1, y1, x2, y2, cx, cy = box

#             dist = np.sqrt((tx - cx)**2 + (ty - cy)**2)

#             if dist < min_dist:
#                 min_dist = dist
#                 best_box = box

#         if best_box is None:
#             continue

#         x1, y1, x2, y2, cx, cy = best_box

#         # -----------------------------
#         # SPATIAL COUNTING
#         # -----------------------------
#         is_new = True
#         for px, py in current_positions:
#             if np.sqrt((cx - px)**2 + (cy - py)**2) < threshold:
#                 is_new = False
#                 break

#         if is_new:
#             current_positions.append((cx, cy))

#         # -----------------------------
#         # DRAW YOLO BOX (CORRECT)
#         # -----------------------------
#         cv2.rectangle(frame,
#                       (int(x1), int(y1)),
#                       (int(x2), int(y2)),
#                       (0, 255, 0), 2)

#         # Draw ID
#         cv2.putText(frame,
#                     f"ID: {track_id}",
#                     (int(x1), int(y1) - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6, (0, 255, 0), 2)

#     # -----------------------------
#     # CURRENT COUNT ONLY
#     # -----------------------------
#     total_now = len(current_positions)

#     cv2.putText(frame,
#                 f"Aircraft Now: {total_now}",
#                 (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8, (0, 0, 255), 2)

#     # -----------------------------
#     # DISPLAY
#     # -----------------------------
#     display = cv2.resize(frame, (1200, 700))
#     cv2.imshow("Aircraft Detection (Final)", display)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO("Models\\aircraft_detector_v8.pt")

tracker = DeepSort(max_age=10, n_init=3)

cap = cv2.VideoCapture("Simulation Videos\\fc_simulation-2.mp4")

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------
    # YOLO DETECTION
    # -----------------------------
    results = model(frame, conf=0.15, imgsz=960)[0]

    detections = []
    current_positions = []

    if results.boxes is not None:
        for r in results.boxes.data:
            x1, y1, x2, y2, conf, cls = r.tolist()

            w = x2 - x1
            h = y2 - y1

            detections.append(([x1, y1, w, h], conf, 'aircraft'))

            # center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            current_positions.append((cx, cy))

            # 🔥 DRAW YOLO BOX DIRECTLY
            cv2.rectangle(frame,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0,255,0), 2)

    # -----------------------------
    # TRACKING (ONLY FOR ID)
    # -----------------------------
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        if track.hits < 3:
            continue

        l, t, w, h = track.to_ltrb()
        track_id = track.track_id

        cx = int(l + w/2)
        cy = int(t + h/2)

        # draw ID near center
        cv2.putText(frame,
                    f"ID: {track_id}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0,255), 2)

    # -----------------------------
    # COUNT (PURE YOLO)
    # -----------------------------
    total_now = len(current_positions)

    cv2.putText(frame,
                f"Aircraft Now: {total_now}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255,0,0), 2)

    # -----------------------------
    # DISPLAY
    # -----------------------------
    display = cv2.resize(frame, (1200, 700))
    cv2.imshow("Final Clean System", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()