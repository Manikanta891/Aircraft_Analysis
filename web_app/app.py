from flask import Flask, request, jsonify
from flask_cors import CORS
import os, math, base64, uuid
import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

app = Flask(__name__)
CORS(app)

# ─── CONFIG ───────────────────────────────────────────────────
MODEL_PATH     = "../Models/aircraft_detector_v11.pt"
UPLOAD_FOLDER  = "uploads"
OUTPUT_FOLDER  = "outputs"
CONF_THRESHOLD = 0.40

COLOR_SMALL  = (255, 100,  50)
COLOR_MEDIUM = ( 50, 220,  50)
COLOR_LARGE  = ( 50,  50, 255)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"Loading model from {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")


from flask import render_template

@app.route("/")
def home():
    return render_template("index.html")

def encode_image_to_base64(img_array):
    success, buffer = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise RuntimeError("Failed to encode image to JPEG")
    return base64.b64encode(buffer).decode("utf-8")


def analyse_image(image_path, output_path):
    yolo_results = model(image_path, verbose=False)
    img          = cv2.imread(image_path)

    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    img_h, img_w = img.shape[:2]
    img_area     = img_h * img_w
    img_diagonal = math.sqrt(img_w ** 2 + img_h ** 2)
    img_mp       = img_area / 1_000_000

    centers, areas, conf_scores, box_coords = [], [], [], []

    for box in yolo_results[0].boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        centers.append([(x1 + x2) // 2, (y1 + y2) // 2])
        areas.append((x2 - x1) * (y2 - y1))
        conf_scores.append(round(conf, 3))
        box_coords.append([x1, y1, x2, y2])

    aircraft_count = len(centers)

    if aircraft_count == 0:
        cv2.putText(img, "No Aircraft Detected", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imwrite(output_path, img)
        return {
            "aircraft_count": 0, "small_aircraft": 0, "medium_aircraft": 0,
            "large_aircraft": 0, "density": "Low", "density_value": 0.0,
            "activity_score": 0, "activity_level": "No Activity",
            "num_clusters": 0, "congestion_level": "No Aircraft",
            "spatial_spread": 0.0, "avg_confidence": 0.0,
            "aircraft_details": [],
            "result_image": encode_image_to_base64(img)
        }

    centers   = np.array(centers,   dtype=np.float32)
    areas_arr = np.array(areas,     dtype=np.float32)

    # Size classification
    size_labels = []
    small = medium = large = 0
    for area in areas_arr:
        ratio = area / img_area
        if ratio < 0.003:
            size_labels.append("small");  small  += 1
        elif ratio < 0.010:
            size_labels.append("medium"); medium += 1
        else:
            size_labels.append("large");  large  += 1

    # Draw bounding boxes
    for i in range(aircraft_count):
        x1, y1, x2, y2 = box_coords[i]
        sz  = size_labels[i]
        col = COLOR_SMALL if sz == "small" else COLOR_MEDIUM if sz == "medium" else COLOR_LARGE
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.circle(img, (int(centers[i][0]), int(centers[i][1])), 3, (0, 255, 255), -1)
        label     = f"{conf_scores[i]:.2f} {sz[0].upper()}"
        label_pos = (x1, min(y2 + 14, img_h - 4)) if y1 < 30 else (x1, y1 - 5)
        cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
        cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    # Clustering
    cluster_labels_list = [0] * aircraft_count
    num_clusters        = 1
    if aircraft_count >= 2:
        eps = 0.12 * img_diagonal
        db  = DBSCAN(eps=eps, min_samples=1).fit(centers)
        cluster_labels_list = db.labels_.tolist()
        num_clusters        = len(set(cluster_labels_list))

    # Density
    density_per_mp = aircraft_count / img_mp if img_mp > 0 else 0
    density = "Low" if density_per_mp < 5.0 else "Medium" if density_per_mp < 15.0 else "High"

    # Congestion
    pair_distances = [
        float(np.linalg.norm(centers[i] - centers[j]))
        for i in range(aircraft_count) for j in range(i + 1, aircraft_count)
    ]
    if pair_distances:
        min_dist_norm    = np.min(pair_distances) / img_diagonal
        congestion_level = ("High Congestion"   if min_dist_norm < 0.04 else
                            "Medium Congestion" if min_dist_norm < 0.10 else
                            "Low Congestion")
    else:
        congestion_level = "Single Aircraft"

    # Spatial spread
    spread_score = 0.0
    if aircraft_count >= 3:
        try:
            hull         = ConvexHull(centers.astype(np.float64))
            spread_score = min(1.0, hull.volume / img_area)
        except Exception:
            spread_score = 0.0
    elif aircraft_count == 2:
        spread_score = min(1.0, float(np.linalg.norm(centers[0] - centers[1])) / img_diagonal)

    # Activity score
    avg_conf      = float(np.mean(conf_scores))
    count_score   = min(1.0, aircraft_count / 40.0)
    density_score = min(1.0, density_per_mp  / 20.0)
    size_variety  = len([x for x in [small, medium, large] if x > 0])
    size_score    = size_variety / 3.0
    conf_score    = min(1.0, avg_conf)

    activity_score = int(round(100 * (
        0.35 * count_score   + 0.25 * density_score +
        0.15 * size_score    + 0.15 * spread_score  + 0.10 * conf_score
    )))
    activity_level = ("Low Activity"      if activity_score < 30 else
                      "Moderate Activity" if activity_score < 55 else
                      "High Activity"     if activity_score < 75 else
                      "Very High Activity")

    # Legend
    lx, ly = 12, img_h - 86
    cv2.rectangle(img, (lx - 4, ly - 14), (lx + 120, ly + 72), (20, 20, 20), -1)
    cv2.rectangle(img, (lx - 4, ly - 14), (lx + 120, ly + 72), (100, 100, 100), 1)
    for idx, (lbl, col) in enumerate([("Small", COLOR_SMALL), ("Medium", COLOR_MEDIUM), ("Large", COLOR_LARGE)]):
        yy = ly + idx * 24
        cv2.rectangle(img, (lx, yy), (lx + 14, yy + 12), col, -1)
        cv2.putText(img, lbl, (lx + 20, yy + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # Stats overlay
    for text, pos, scale in [
        (f"Aircraft: {aircraft_count}",                     (30,  38), 0.80),
        (f"Density: {density} ({density_per_mp:.1f}/MP)",   (30,  72), 0.70),
        (f"Activity: {activity_level} ({activity_score})",  (30, 106), 0.70),
        (f"Clusters: {num_clusters}  |  {congestion_level}",(30, 138), 0.60),
        (f"S:{small}  M:{medium}  L:{large}",               (30, 168), 0.68),
    ]:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)

    # Save to disk
    cv2.imwrite(output_path, img)
    print(f"Output saved -> {output_path}")

    # Encode directly from array — no file read needed, no path issues
    result_b64 = encode_image_to_base64(img)

    aircraft_details = [{
        "id": i + 1, "bbox": box_coords[i],
        "center": [int(centers[i][0]), int(centers[i][1])],
        "size_class": size_labels[i], "area_px": int(areas_arr[i]),
        "confidence": conf_scores[i], "cluster_id": int(cluster_labels_list[i])
    } for i in range(aircraft_count)]

    return {
        "aircraft_count":   int(aircraft_count),
        "small_aircraft":   int(small),
        "medium_aircraft":  int(medium),
        "large_aircraft":   int(large),
        "density":          str(density),
        "density_value":    round(density_per_mp, 4),
        "activity_score":   int(activity_score),
        "activity_level":   str(activity_level),
        "num_clusters":     int(num_clusters),
        "congestion_level": str(congestion_level),
        "spatial_spread":   round(float(spread_score), 4),
        "avg_confidence":   round(avg_conf, 4),
        "aircraft_details": aircraft_details,
        "result_image":     result_b64
    }


@app.route("/analyse", methods=["POST"])
def analyse():
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext         = os.path.splitext(file.filename)[1] or ".jpg"
    safe_name   = str(uuid.uuid4()) + ext
    input_path  = os.path.join(UPLOAD_FOLDER, safe_name)
    output_path = os.path.join(OUTPUT_FOLDER, "result_" + safe_name)
    file.save(input_path)
    print(f"Received: {file.filename} -> {safe_name}")

    try:
        data = analyse_image(input_path, output_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    return jsonify(data)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)