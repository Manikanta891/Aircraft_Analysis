# TEST CASES DOCUMENTATION

## SMART AIRPORT MONITORING SYSTEM

---

## 1. Aircraft Detection Tests

| Test ID | Test Case | Expected Result | Status |
|---------|-----------|----------------|--------|
| TC-01 | Single aircraft in video frame | Aircraft detected with bounding box | PASS |
| TC-02 | Multiple aircraft in video frame | All aircraft detected with unique IDs | PASS |
| TC-03 | No aircraft in video frame | "No Aircraft Detected" message displayed | PASS |
| TC-04 | Aircraft at far distance (small) | Small aircraft classified correctly | PASS |

---

## 2. Runway Collision Detection Tests

| Test ID | Test Case | Expected Result | Status |
|---------|-----------|----------------|--------|
| TC-05 | Two aircraft moving toward each other on same runway | HIGH_RISK alert triggered after 4 consecutive frames | PASS |
| TC-06 | Two aircraft moving away from each other | SAFE status displayed | PASS |
| TC-07 | Aircraft on different runways | No collision alert generated | PASS |
| TC-08 | Single aircraft on runway | ACTIVE status, no collision warning | PASS |
| TC-09 | Aircraft static on runway | STATIC status displayed | PASS |

---

## 3. Restricted Zone Monitoring Tests

| Test ID | Test Case | Expected Result | Status |
|---------|-----------|----------------|--------|
| TC-10 | Aircraft enters restricted zone | Zone violation detected and logged | PASS |
| TC-11 | Aircraft stays in restricted zone for more than 5 seconds | WARNING alert generated | PASS |
| TC-12 | Aircraft stays in restricted zone for more than 10 seconds | CRITICAL alert generated | PASS |
| TC-13 | Multiple aircraft detected in restricted zone | VIOLATED status displayed | PASS |

---

## 4. Parking Slot Monitoring Tests

| Test ID | Test Case | Expected Result | Status |
|---------|-----------|----------------|--------|
| TC-14 | Aircraft parks in an empty slot | Slot shows OCCUPIED status with aircraft ID | PASS |
| TC-15 | Aircraft departs from occupied slot | Slot shows FREE status, duration logged | PASS |
| TC-16 | Aircraft stays in parking slot for more than 60 seconds | WARNING alert generated | PASS |
| TC-17 | Aircraft stays in parking slot for more than 120 seconds | CRITICAL alert generated | PASS |

---

## 5. Terminal Gate Occupancy Tests

| Test ID | Test Case | Expected Result | Status |
|---------|-----------|----------------|--------|
| TC-18 | Aircraft docks at gate | Gate shows OCCUPIED status | PASS |
| TC-19 | Aircraft departs from gate | Gate shows FREE status, departure logged | PASS |
| TC-20 | Multiple aircraft detected at same gate | ERROR status displayed | PASS |
| TC-21 | Aircraft overstays at terminal gate | DELAY warning displayed | PASS |

---

## 6. API & Dashboard Tests

| Test ID | Test Case | Expected Result | Status |
|---------|-----------|----------------|--------|
| TC-22 | Click Start button on dashboard | Video stream begins, status changes to LIVE | PASS |
| TC-23 | Click Stop button on dashboard | Stream stops, status changes to STOPPED | PASS |
| TC-24 | Fetch /api/data endpoint | Valid JSON with runway/aircraft data returned | PASS |
| TC-25 | Video reaches end and loops | System resets, processing continues | PASS |

---

## 7. Performance Tests

| Test ID | Test Case | Expected Result | Status |
|---------|-----------|----------------|--------|
| TC-26 | System runs with single aircraft | FPS maintains 25-30 | PASS |
| TC-27 | System runs with 5+ aircraft | FPS maintains 15-25 | PASS |
| TC-28 | YOLO model inference time | Less than 50ms per frame | PASS |

---

## 8. Summary

| Category | Total Test Cases | Passed | Failed |
|----------|------------------|--------|--------|
| Aircraft Detection | 4 | 4 | 0 |
| Runway Collision | 5 | 5 | 0 |
| Restricted Zone | 4 | 4 | 0 |
| Parking Slot | 4 | 4 | 0 |
| Terminal Gate | 4 | 4 | 0 |
| API & Dashboard | 4 | 4 | 0 |
| Performance | 3 | 3 | 0 |
| **TOTAL** | **28** | **28** | **0** |

---

## Test Environment

| Component | Specification |
|-----------|---------------|
| Processor | Intel Core i5/i7 |
| RAM | 8 GB minimum |
| GPU | NVIDIA with CUDA support |
| Operating System | Windows 10/11 |
| Python Version | 3.10 |
| YOLO Model | v11 (aircraft_detector_v11.pt) |
| DeepSort | deep_sort_realtime |

---

## Test Data

| Video File | Purpose | Duration |
|------------|---------|----------|
| runway_simulation.mp4 | Collision detection testing | 30 seconds |
| restricted_video.mp4 | Zone monitoring testing | 20 seconds |
| parking_simulation.mp4 | Parking slot testing | 25 seconds |
| terminal_simulation.mp4 | Terminal gate testing | 30 seconds |

---

o