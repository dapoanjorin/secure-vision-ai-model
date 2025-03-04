import os
import cv2
import time
import requests
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# 🔹 Flask App Initialization
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# 🔹 YOLOv8 Model Load
print("📌 Loading YOLOv8 model...")
model = YOLO("yolov8s.pt")  # Using YOLOv8 small model
conf_threshold = 0.25  # Confidence threshold

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

    # Define classes of interest with their respective COCO class indices for YOLOv8
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

        # Process frame with YOLOv8
        results = model(frame, conf=conf_threshold)
        
        # YOLOv8 returns a Results object which we can iterate through
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                cls_id = int(box.cls.item())
                
                # Check if detected class is in our classes of interest
                if cls_id in classes_of_interest:
                    class_name = classes_of_interest[cls_id]
                    confidence = box.conf.item()
                    
                    # Get bounding box coordinates and convert to integers
                    x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                    
                    print(f"  - {class_name}: {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")
                    
                    detected_objects.append({
                        "object": class_name,
                        "confidence": confidence,
                        "bounding_box": [x1, y1, x2, y2]
                    })
                    
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