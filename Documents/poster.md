# Smart Airport Monitoring System - Summary

---

## Abstract

This project presents a Smart Airport Monitoring System designed to enhance airport operations through real-time computer vision and deep learning techniques. The system employs YOLOv11 for aircraft detection and DeepSort for multi-object tracking to provide four key monitoring capabilities: runway collision detection, terminal gate occupancy monitoring, parking slot management, and restricted zone surveillance. A Flask-based web dashboard provides real-time visualization through MJPEG streaming, enabling airport personnel to monitor operations remotely. Experimental results demonstrate the system's effectiveness in detecting and tracking aircraft across various airport scenarios, with configurable alerts for safety-critical events.

---

## Methodology

1. **Aircraft Detection (YOLOv11)** - Custom YOLOv11 model detects aircraft with confidence threshold 0.25 and image size 640x640.

2. **Object Tracking (DeepSort)** - Uses max_age=30 and n_init=3 to maintain persistent track IDs across frames.

3. **Collision Detection** - Analyzes trajectory angles; flags HIGH_RISK when angle <30° for 4+ consecutive frames.

4. **Terminal Occupancy** - Point-in-box algorithm monitors gate status (Occupied/Free) using JSON zone definitions.

5. **Parking Slot Monitoring** - Tracks slot availability and duration with long-stay alerts (60s warning, 120s critical).

6. **Restricted Zone Monitoring** - Two-tier severity based on time inside: Normal (<5s), Warning (5-10s), Critical (>10s).

7. **Web Dashboard** - Flask with MJPEG streaming and thread-safe JSON API polling every 1.5 seconds.

---

## Conclusion

This project successfully developed a Smart Airport Monitoring System that integrates YOLOv11 object detection with DeepSort multi-object tracking to provide four distinct monitoring capabilities: collision detection, terminal occupancy monitoring, parking slot management, and restricted zone surveillance.

Key achievements include:
- Real-time aircraft detection with high accuracy
- Stable multi-object tracking with consistent track IDs
- Functional web dashboards with MJPEG streaming
- Modular architecture enabling easy extension

Future work includes real-world deployment testing, multi-camera integration, and integration with existing airport notification systems.

---