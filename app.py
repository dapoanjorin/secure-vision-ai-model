import os
import cv2
import time
import requests
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Flask App Initialization
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# YOLOv8 Model Load
print("Loading YOLOv8 model...")
model = YOLO("yolov8s.pt")
conf_threshold = 0.25

# Server URL for Alert
ALERT_API_URL = "https://secure-vision-server.onrender.com/api/alerts"

# Classes of interest for security monitoring (COCO dataset indices)
CLASSES_OF_INTEREST = {
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
    56: "chair",
    57: "couch",
    63: "laptop",
    67: "cell phone",
    73: "scissors"
}

# High alert objects that require immediate notifications
HIGH_ALERT_OBJECTS = [
    "knife", "scissors", "sports ball", "bottle", "handbag", "suitcase",
    "cell phone", "laptop"
]

def get_threat_severity(object_name):
    """Determine threat severity based on object type."""
    high_severity_objects = ["knife", "scissors", "baseball bat"]
    medium_severity_objects = ["bottle", "sports ball", "backpack", "suitcase", "handbag"]
    
    if object_name in high_severity_objects:
        return "High"
    elif object_name in medium_severity_objects:
        return "Medium"
    else:
        return "Low"

def send_alert(object_name, confidence, bounding_box, location="Main Gate", frame_timestamp=None):
    """Send security alert to the API server."""
    severity = get_threat_severity(object_name)
    
    alert_data = {
        "type": "Security Alert",
        "description": f"Detected {object_name} with {confidence:.2f} confidence",
        "severity": severity,
        "location": location,
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
            print(f"Alert sent successfully for {object_name} ({severity} severity).")
        else:
            print(f"Failed to send alert. Status: {response.status_code}")
    except Exception as e:
        print(f"Error sending alert: {str(e)}")

def process_frame(frame, frame_count=0, location="Unknown", frame_timestamp=None):
    """Process a single frame with YOLOv8 model and detect objects of interest."""
    if frame_timestamp is None:
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
            if cls_id in CLASSES_OF_INTEREST:
                class_name = CLASSES_OF_INTEREST[cls_id]
                confidence = box.conf.item()
                
                # Get bounding box coordinates and convert to integers
                x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                
                detection_info = {
                    "object": class_name,
                    "confidence": confidence,
                    "bounding_box": [x1, y1, x2, y2],
                    "frame": frame_count,
                    "timestamp": frame_timestamp,
                    "location": location
                }
                
                frame_detections.append(detection_info)
                
                # Send an alert for potential threat objects
                if class_name in HIGH_ALERT_OBJECTS:
                    send_alert(class_name, confidence, [x1, y1, x2, y2], location, frame_timestamp)
    
    return frame_detections

def process_video(video_path, location="Main Gate"):
    """Process a complete video file and detect objects of interest."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}

    print(f"Processing video from {location}...")
    frame_count = 0
    
    detected_objects = []
    suspicious_behavior_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        frame_detections = process_frame(frame, frame_count, location, frame_timestamp)
        detected_objects.extend(frame_detections)
        
        # Simple suspicious behavior detection: multiple people and potential weapons
        person_count = sum(1 for d in frame_detections if d["object"] == "person")
        weapon_count = sum(1 for d in frame_detections if d["object"] in ["knife", "scissors"])
        
        if person_count >= 2 and weapon_count >= 1:
            suspicious_behavior_count += 1
            if suspicious_behavior_count >= 5:  # If sustained for multiple frames
                print(f"SUSPICIOUS BEHAVIOR DETECTED at {location}: Multiple people with potential weapons")
                # Special high-priority alert
                alert_data = {
                    "type": "Security Alert",
                    "description": f"SUSPICIOUS BEHAVIOR: {person_count} people with {weapon_count} potential weapons",
                    "severity": "Critical",
                    "location": location,
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
                    print(f"Error sending behavior alert: {str(e)}")

    cap.release()
    return {
        "detections": detected_objects, 
        "message": f"Video processing complete for {location}",
        "statistics": {
            "frames_processed": frame_count,
            "objects_detected": len(detected_objects),
            "potential_threats": sum(1 for d in detected_objects if d["object"] in HIGH_ALERT_OBJECTS)
        }
    }

@app.route("/detect", methods=["POST"])
def detect():
    """API endpoint to upload and process a video file."""
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
        
    location = request.form.get("location", "Main Gate")

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    print(f"Video saved to: {file_path}")

    # Process the video
    result = process_video(file_path, location)

    # Remove the file after processing
    os.remove(file_path)

    return jsonify(result), 200

@app.route("/health", methods=["GET"])
def health_check():
    """API endpoint to check service health."""
    return jsonify({
        "status": "healthy",
        "model": "YOLOv8s",
        "version": "1.0"
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)