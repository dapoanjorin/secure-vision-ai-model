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

# 🔹 Define threat severity levels
def get_threat_severity(object_name):
    high_severity_objects = ["knife", "scissors", "baseball bat", "gun", "rifle", "pistol", 
                            "firearm", "axe", "hammer", "crowbar"]
    medium_severity_objects = ["bottle", "sports ball", "backpack", "suitcase", "handbag"]
    
    if object_name in high_severity_objects:
        return "High"
    elif object_name in medium_severity_objects:
        return "Medium"
    else:
        return "Low"

# 🔹 Function to Send Alerts
def send_alert(object_name, confidence, bounding_box, frame_timestamp=None):
    severity = get_threat_severity(object_name)
    
    alert_data = {
        "type": "Security Alert",
        "description": f"Detected {object_name} with {confidence:.2f} confidence",
        "severity": severity,
        "location": "Main Gate",
        "timestamp": frame_timestamp or time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": {
            "object": object_name,
            "confidence": confidence,
            "bounding_box": bounding_box,
            "threat_level": severity
        },
    }

    try:
        response = requests.post(ALERT_API_URL, json=alert_data, headers={"Content-Type": "application/json"})
        if response.status_code == 201:
            print(f"✅ Alert sent successfully for {object_name} ({severity} severity).")
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
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Expanded list of classes of interest for security monitoring
    # COCO dataset class indices (https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/)
    classes_of_interest = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        32: "sports ball",
        39: "bottle",
        41: "cup",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        56: "chair",
        57: "couch",
        59: "potted plant",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        73: "scissors",
        74: "teddy bear",
        76: "scissors",
        78: "microwave",
        79: "oven",
        82: "refrigerator"
    }
    
    # Additional objects that would require alerts
    high_alert_objects = [
        "knife", "scissors", "sports ball", "bottle", "handbag", "suitcase",
        "cell phone", "laptop"
    ]

    detected_objects = []
    suspicious_behavior_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Process frame with YOLOv8
        results = model(frame, conf=conf_threshold)
        
        frame_detections = []
        
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
                    
                    detection_info = {
                        "object": class_name,
                        "confidence": confidence,
                        "bounding_box": [x1, y1, x2, y2],
                        "frame": frame_count,
                        "timestamp": frame_timestamp
                    }
                    
                    frame_detections.append(detection_info)
                    detected_objects.append(detection_info)
                    
                    print(f"  - {class_name}: {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")
                    
                    # Send an alert for potential threat objects
                    if class_name in high_alert_objects:
                        send_alert(class_name, confidence, [x1, y1, x2, y2], frame_timestamp)
        
        # Simple suspicious behavior detection: multiple people and potential weapons
        person_count = sum(1 for d in frame_detections if d["object"] == "person")
        weapon_count = sum(1 for d in frame_detections if d["object"] in ["knife", "scissors"])
        
        if person_count >= 2 and weapon_count >= 1:
            suspicious_behavior_count += 1
            if suspicious_behavior_count >= 5:  # If sustained for multiple frames
                print("⚠️ SUSPICIOUS BEHAVIOR DETECTED: Multiple people with potential weapons")
                # Special high-priority alert
                alert_data = {
                    "type": "Security Alert",
                    "description": f"SUSPICIOUS BEHAVIOR: {person_count} people with {weapon_count} potential weapons",
                    "severity": "Critical",
                    "location": "Main Gate",
                    "timestamp": frame_timestamp,
                    "metadata": {
                        "behavior": "group_with_weapons",
                        "person_count": person_count,
                        "weapon_count": weapon_count,
                        "frame": frame_count
                    },
                }
                try:
                    requests.post(ALERT_API_URL, json=alert_data, headers={"Content-Type": "application/json"})
                except Exception as e:
                    print(f"❌ Error sending behavior alert: {str(e)}")

    cap.release()
    return {
        "detections": detected_objects, 
        "message": "Video processing complete",
        "statistics": {
            "frames_processed": frame_count,
            "objects_detected": len(detected_objects),
            "potential_threats": sum(1 for d in detected_objects if d["object"] in high_alert_objects)
        }
    }

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

# 🔹 Health Check Endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model": "YOLOv8s",
        "version": "1.0"
    }), 200

# 🔹 Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)