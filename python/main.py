from flask import Flask, request, jsonify
import cv2
import time
from datetime import datetime
import os
import warnings
import numpy as np
import requests
import threading
import json
from supervision import Detections
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
import psutil

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Global variables
camera_active = False
camera_thread = None
interval_time = 60  # Default interval in seconds
last_detection_time = None
model = None
cap = None
weather_data = None
next_interval_time = None

# Firebase setup - keep JSON method, remove storage bucket
FIREBASE_CREDENTIALS = os.environ.get("FIREBASE_CREDENTIALS", "../firebase/firebase-credentials.json")
# Remove storage bucket reference

try:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    # Initialize without storage bucket
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    firebase_enabled = True
except Exception as e:
    print(f"Firebase initialization error: {e}")
    firebase_enabled = False

# Weather API configuration 
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "22ba524647a0d39172ebc63307bbf2f1")

LAT = float(os.environ.get("WEATHER_LAT", "37.7749"))  # Default latitude
LON = float(os.environ.get("WEATHER_LON", "-122.4194"))  # Default longitude

# Keep utility functions from original code
def draw_central_box(frame, box_size=50):
    """Draws a central box on the frame."""
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    top_left = (center_x - box_size // 2, center_y - box_size // 2)
    bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    return center_x, center_y

def draw_pixel_grid_with_labels(frame, step=50):
    """Draws a pixel grid across the entire frame and labels intersections."""
    height, width = frame.shape[:2]
    
    for x in range(0, width, step):
        cv2.line(frame, (x, 0), (x, height), (0, 0, 255), 1)
        for y in range(0, height, step):
            cv2.line(frame, (0, y), (width, y), (0, 0, 255), 1)
            
            label = f"({x},{y})"
            cv2.putText(frame, label, (x + 5, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

def load_yolo_model(model_path):
    """Load YOLO model with error handling"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = YOLO(model_path)
        print(f"YOLO model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"YOLO Loading Error: {str(e)}")
        return None

def calculate_distance(center_x, center_y, bbox):
    """Calculates the distance of the detected object from the center."""
    x1, y1, x2, y2 = bbox
    object_center_x = (x1 + x2) / 2
    object_center_y = (y1 + y2) / 2
    distance_x = object_center_x - center_x
    distance_y = object_center_y - center_y
    return distance_x, distance_y

def yolo_to_detections(yolo_result):
    """Convert YOLO results to Supervision Detections format"""
    try:
        if yolo_result is None or len(yolo_result.boxes) == 0:
            return Detections.empty()
            
        boxes = yolo_result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confidence = boxes.conf.cpu().numpy()
        class_id = boxes.cls.cpu().numpy().astype(int)
        
        return Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
    except Exception as e:
        print(f"Detection conversion error: {e}")
        return Detections.empty()

# Weather and interval management
def get_weather_data():
    """Fetch current weather data using a weather API"""
    global weather_data
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={WEATHER_API_KEY}"
        response = requests.get(url)
        data = response.json()
        weather_data = {
            "weather_condition": data["weather"][0]["main"],
            "weather_description": data["weather"][0]["description"],
            "temperature": data["main"]["temp"],
            "clouds": data["clouds"]["all"],
            "wind_speed": data["wind"]["speed"],
            "sunrise": data["sys"]["sunrise"],
            "sunset": data["sys"]["sunset"],
            "timestamp": datetime.now().isoformat()
        }
        return weather_data
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

def calculate_next_interval():
    """Calculate the next interval time based on weather conditions and time of day"""
    global interval_time, next_interval_time
    
    if weather_data is None:
        get_weather_data()
    
    if weather_data:
        # Get current time
        current_time = time.time()
        sunrise = weather_data.get("sunrise")
        sunset = weather_data.get("sunset")
        
        # Check if it's nighttime (after sunset or before sunrise)
        is_nighttime = current_time > sunset or current_time < sunrise
        
        if is_nighttime:
            # If sun is set, use a very long interval until next sunrise
            time_until_sunrise = sunrise - current_time if current_time < sunrise else (sunrise + 86400) - current_time
            new_interval = min(int(time_until_sunrise), 3600)  # Cap at 1 hour max
            interval_formula = "Nighttime - sun is set, waiting until sunrise"
        else:
            # Base interval on weather conditions during daytime
            weather_condition = weather_data["weather_condition"].lower()
            cloud_coverage = weather_data.get("clouds", 0)
            
            # Adjust interval based on weather conditions
            if "clear" in weather_condition or cloud_coverage < 20:
                # Clear sky or minimal clouds: shorter interval
                new_interval = 60  # 1 minute
            elif "cloud" in weather_condition or cloud_coverage < 70:
                # Partly cloudy: medium interval
                new_interval = 180  # 3 minutes
            else:
                # Overcast or rainy: longer interval
                new_interval = 300  # 5 minutes
            
            interval_formula = f"Daytime - Based on {weather_condition} with {cloud_coverage}% cloud coverage"
        
        # Update the interval time
        interval_time = new_interval
        next_interval_time = datetime.now().timestamp() + interval_time
        
        # Log the interval calculation to Firebase
        post_program_details_to_firebase(weather_data, interval_formula, next_interval_time)
        
        return interval_time
    
    # Default interval if weather data is not available
    interval_time = 120  # 2 minutes
    next_interval_time = datetime.now().timestamp() + interval_time
    return interval_time

# Image and video processing functions
def process_image_with_model(image, return_annotated=False):
    """Process an image with the YOLO model and return results"""
    try:
        if model is None:
            return {"error": "Model not loaded"}, None, None
            
        # Create a copy of the image for processing
        frame = image.copy()
        
        # Draw central box on the frame
        center_x, center_y = draw_central_box(frame)
        
        # Process image with YOLO
        results = model.predict(source=frame, conf=0.3, verbose=False)[0]
        
        # Process results
        detections = []
        if results is not None and results.boxes:
            for box, cls_id, conf in zip(
                results.boxes.xyxy.cpu().numpy(), 
                results.boxes.cls.cpu().numpy().astype(int),
                results.boxes.conf.cpu().numpy()
            ):
                # Process only class_0 (sun)
                if cls_id == 0:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Calculate distances from the center
                    distance_x, distance_y = calculate_distance(center_x, center_y, (x1, y1, x2, y2))
                    
                    # Add to detections list
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                        "distance_x": float(distance_x),
                        "distance_y": float(distance_y)
                    })
                    
                    # Draw bounding box and distance info on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        f"dx: {distance_x:.1f}, dy: {distance_y:.1f}",
                        (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1
                    )
        
        # Prepare response
        response = {
            "detections": detections,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save the processed image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs("results", exist_ok=True)
        output_path = f"results/api_output_{timestamp}.jpg"
        cv2.imwrite(output_path, frame)
        
        if return_annotated:
            return response, frame, output_path
        return response, None, output_path
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return {"error": str(e)}, None, None

# Remove the entire get_model_response() endpoint function

# Modify the camera function to run directly instead of in a thread
def camera_function():
    """Function to run the camera and model detection"""
    global camera_active, cap, last_detection_time, next_interval_time
    
    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            camera_active = False
            return
        
        # Initial calculations
        calculate_next_interval()
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        print("Camera started, beginning detection loop")
        camera_active = True
        
        while camera_active:
            # Check if it's time to capture and process
            current_time = datetime.now().timestamp()
            
            if next_interval_time is None or current_time >= next_interval_time:
                print(f"Processing frame at {datetime.now().isoformat()}")
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    time.sleep(1)
                    continue
                
                # Process the frame
                results, annotated_frame, output_path = process_image_with_model(frame, return_annotated=True)
                
                # Log results to Firebase using the internal function
                if "error" not in results:
                    # Get system info
                    system_info = {
                        "cpu_percent": psutil.cpu_percent(),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_percent": psutil.disk_usage('/').percent
                    }
                    
                    # Log model status using internal function
                    post_current_status_to_firebase(
                        model_details={
                            "detections": results["detections"],
                            "timestamp": results["timestamp"]
                        },
                        raspberry_details=system_info
                    )
                
                # Calculate next interval
                get_weather_data()  # Update weather data
                calculate_next_interval()
                
                last_detection_time = current_time
            
            # Sleep for a short time to avoid high CPU usage
            time.sleep(1)
        
    except Exception as e:
        print(f"Camera function error: {e}")
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        camera_active = False
        print("Camera stopped")
        
# Flask API Endpoints
@app.route('/start_stop_camera', methods=['PUT'])
def start_stop_camera():
    """Endpoint to start or stop the camera and model detection"""
    global camera_active
    
    try:
        data = request.json
        action = data.get('action', '').lower()
        
        if action == 'start' and not camera_active:
            # Initialize model if not loaded
            if model is None:
                return jsonify({"error": "Model not loaded"}), 500
            
            camera_thread = threading.Thread(target=camera_function)
            camera_thread.daemon = True
            camera_thread.start()
            
            return jsonify({
                "status": "success",
                "message": "Camera and model detection started",
                "timestamp": datetime.now().isoformat()
            })
            
        elif action == 'stop' and camera_active:
            # Signal the camera function to stop
            camera_active = False
            
            return jsonify({
                "status": "success",
                "message": "Camera and model detection stopped",
                "timestamp": datetime.now().isoformat()
            })
            
        else:
            return jsonify({
                "status": "error",
                "message": f"Invalid action '{action}' or camera already in requested state",
                "camera_active": camera_active,
                "timestamp": datetime.now().isoformat()
            }), 400
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/change_interval', methods=['PUT'])
def change_interval():
    """Endpoint to change the interval time for camera operation"""
    global interval_time, next_interval_time
    
    try:
        data = request.json
        new_interval = data.get('interval')
        
        if new_interval is None:
            return jsonify({
                "status": "error",
                "message": "Missing 'interval' parameter",
                "timestamp": datetime.now().isoformat()
            }), 400
            
        try:
            new_interval = int(new_interval)
            if new_interval <= 0:
                raise ValueError("Interval must be positive")
        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "Invalid interval value. Must be a positive integer.",
                "timestamp": datetime.now().isoformat()
            }), 400
            
        # Update interval time
        old_interval = interval_time
        interval_time = new_interval
        next_interval_time = datetime.now().timestamp() + interval_time
        
        # Log to Firebase using internal function
        post_program_details_to_firebase(
            weather_response=weather_data,
            interval_formula=f"Interval changed manually from {old_interval}s to {interval_time}s",
            next_interval_time=next_interval_time
        )
        
        return jsonify({
            "status": "success",
            "message": f"Interval updated from {old_interval} to {interval_time} seconds",
            "next_interval_time": next_interval_time,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


# Remove these API endpoints and convert to internal functions
def post_current_status_to_firebase(model_details, raspberry_details=None):
    """Internal function to log current model and Raspberry Pi status to Firebase"""
    try:
        if not firebase_enabled:
            return False
            
        # Get system info if not provided
        if raspberry_details is None:
            raspberry_details = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
            
        # Prepare data
        log_data = {
            "model_details": model_details,
            "raspberry_details": raspberry_details,
            "timestamp": datetime.now().isoformat()
        }
            
        # Log to Firebase
        db.collection("ModelLog").add(log_data)
        return True
        
    except Exception as e:
        print(f"Error logging model status to Firebase: {e}")
        return False

def post_program_details_to_firebase(weather_response, interval_formula, next_interval_time):
    """Internal function to log program details to Firebase"""
    try:
        if not firebase_enabled:
            return False
            
        # Prepare data
        log_data = {
            "weather_response": weather_response,
            "interval_formula": interval_formula,
            "next_interval_time": next_interval_time,
            "timestamp": datetime.now().isoformat()
        }
            
        # Log to Firebase
        db.collection("ProgramLog").add(log_data)
        return True
        
    except Exception as e:
        print(f"Error logging program details to Firebase: {e}")
        return False

@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint to get the current status of the system"""
    try:
        status = {
            "camera_active": camera_active,
            "interval_time": interval_time,
            "next_interval_time": next_interval_time,
            "last_detection_time": last_detection_time,
            "model_loaded": model is not None,
            "weather_data": weather_data,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

# Initialize the application
def initialize():
    """Initialize the application, load model, etc."""
    global model
    
    # Specify the default model path
    model_path = os.environ.get("MODEL_PATH", "../models/sun_tracker_v3/sun_tracker_v3_float32.tflite")
    
    # Load the YOLO model
    model = load_yolo_model(model_path)
    
    if model is None:
        print("Warning: Failed to load the model. Endpoints requiring model will not work.")
    
    # Get initial weather data
    get_weather_data()
    
    print("Initialization complete")

# Run the application
if __name__ == '__main__':
    initialize()
    app.run(host='0.0.0.0', port=5000, debug=True)
