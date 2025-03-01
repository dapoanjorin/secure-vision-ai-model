import os
import cv2
import torch
import time
import requests
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# 🔹 Flask App Initialization
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# 🔹 YOLOv5 Model Load
print("📌 Loading YOLOv5 model...")
model = torch.hub.load("ultralytics/yolov5", "yolov5s")
model.conf = 0.25  # Confidence threshold

# 🔹 Server URL for Alert
ALERT_API_URL = "https://secure-vision-server.onrender.com/api/alerts"

# 🔹 Function to Send Alerts
def send_alert(object_name, confidence, bounding_box):
    alert_data = {
        "type": "Security Alert",
        "description": f"Detected {object_name} with {confidence:.2f} confidence",
        "severity": "High",
        "location": "Main Gate",
        "metadata": {
            "object": object_name,
            "confidence": confidence,
            "bounding_box": bounding_box,
        },
    }

    try:
        response = requests.post(ALERT_API_URL, json=alert_data, headers={"Content-Type": "application/json"})
        if response.status_code == 201:
            print("✅ Alert sent successfully.")
        else:
            print(f"❌ Failed to send alert. Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error sending alert: {str(e)}")

# 🔹 Function to Process Video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}

    print("🚀 Processing video...")

    classes_of_interest = {
        0: "person",
        2: "car",
        3: "motorcycle",
        43: "knife",  # Crime-related object
        67: "cell phone",
    }

    detected_objects = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.pandas().xyxy[0]
        relevant_detections = detections[detections["class"].isin(classes_of_interest.keys())]

        if not relevant_detections.empty:
            print("\n🔍 Detected Objects:")
            for _, detection in relevant_detections.iterrows():
                class_id = int(detection["class"])
                class_name = classes_of_interest.get(class_id, f"class_{class_id}")
                confidence = detection["confidence"]
                x1, y1, x2, y2 = map(int, detection[["xmin", "ymin", "xmax", "ymax"]])

                print(f"  - {class_name}: {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")

                detected_objects.append({"object": class_name, "confidence": confidence, "bounding_box": [x1, y1, x2, y2]})

                # Send an alert for a detected weapon
                if class_name == "knife":
                    send_alert(class_name, confidence, [x1, y1, x2, y2])

    cap.release()
    return {"detections": detected_objects, "message": "Video processing complete"}

# 🔹 API Endpoint to Upload & Process Video
@app.route("/detect", methods=["POST"])
def detect():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    print(f"📂 Video saved to: {file_path}")

    # Process the video
    result = process_video(file_path)

    # Remove the file after processing
    os.remove(file_path)

    return jsonify(result), 200

# 🔹 Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
