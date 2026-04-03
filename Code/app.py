# This is a simple Streamlit app that allows users to upload an image and analyzes it for aircraft using a YOLO model. 
# It provides a count of detected aircraft, their density, and an activity level based on the count. 
# The app displays the uploaded image with bounding boxes around detected aircraft and shows the analysis results below the image.

def analyze_aircraft(image_path):

    from ultralytics import YOLO
    import cv2, numpy as np

    model = YOLO("../models/aircraft_detector_v8.pt")

    results = model(image_path, verbose=False)
    img = cv2.imread(image_path)

    boxes = results[0].boxes

    centers = []
    areas = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        centers.append([cx, cy])
        areas.append((x2 - x1)*(y2 - y1))

        cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)

    aircraft_count = len(centers)

    if aircraft_count == 0:
        return {
            "image": img,
            "count": 0,
            "density": "Low",
            "activity": "No Activity"
        }

    centers = np.array(centers)

    # simple density
    density_value = aircraft_count / (img.shape[0]*img.shape[1])

    if density_value < 0.00002:
        density = "Low"
    elif density_value < 0.00005:
        density = "Medium"
    else:
        density = "High"

    # simple activity
    activity_score = min(100, aircraft_count * 5)

    if activity_score < 30:
        activity = "Low"
    elif activity_score < 60:
        activity = "Moderate"
    else:
        activity = "High"

    return {
        "image": img,
        "count": aircraft_count,
        "density": density,
        "activity": activity
    }
    
    
import streamlit as st
import cv2

st.title("✈️ Aircraft Spatial Intelligence System")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file is not None:

    # Save temp image
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    result = analyze_aircraft("temp.jpg")

    st.image(result["image"], channels="BGR")

    st.write("### Results")
    st.write(f"Aircraft Count: {result['count']}")
    st.write(f"Density: {result['density']}")
    st.write(f"Activity Level: {result['activity']}")