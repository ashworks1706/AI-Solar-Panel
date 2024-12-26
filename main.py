from inference import get_model
import supervision as sv
import cv2
import numpy as np
import time
from datetime import datetime, timezone
from suncalc import get_position
import math
from datetime import datetime, timezone, timedelta
import geocoder


def draw_central_box(frame, box_size=100):
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    top_left = (center_x - box_size // 2, center_y - box_size // 2)
    bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    return top_left, bottom_right

def calculate_distance_to_box(sun_center, top_left, bottom_right):
    sun_x, sun_y = sun_center
    box_x_min, box_y_min = top_left
    box_x_max, box_y_max = bottom_right
    
    closest_x = max(box_x_min, min(sun_x, box_x_max))
    closest_y = max(box_y_min, min(sun_y, box_y_max))
    
    return sun_x - closest_x, sun_y - closest_y

def apply_sun_filter(frame):
    img = cv2.convertScaleAbs(frame, alpha=1.2)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.convertScaleAbs(hsv[:,:,2], alpha=-0.40)
    filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return filtered

def calculate_sun_movement(lat, lon, time1, time2):
    pos1 = get_position(time1, lat, lon)
    pos2 = get_position(time2, lat, lon)
    movement = math.sqrt((pos2['azimuth'] - pos1['azimuth'])**2 + (pos2['altitude'] - pos1['altitude'])**2)
    return math.degrees(movement)

def calculate_detection_interval(lat, lon, current_time):
    future_time = current_time + timedelta(minutes=5)
    movement = calculate_sun_movement(lat, lon, current_time, future_time)  # Check movement over 5 minutes
    if movement > 1:  # If sun moves more than 1 degree in 5 minutes
        return 60  # Check every minute
    elif movement > 0.5:  # If sun moves more than 0.5 degrees in 5 minutes
        return 180  # Check every 3 minutes
    else:
        return 300  # Check every 5 minutes

def turn_off_camera(cap):
    cap.release()
    print("Camera turned off.")

def turn_on_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to turn on the camera.")
    else:
        print("Camera turned on.")
    return cap

# Load the pre-trained model
model = get_model(model_id="sun-tracking-555mn/4", api_key=" ")

# Create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Get current location
g = geocoder.ip('me')
latitude, longitude = g.latlng

print(f"Current location: {g.city}, {g.state}, {g.country}")
print(f"Latitude: {latitude}, Longitude: {longitude}")

# Initial sun detection
sun_detected = False
cap = turn_on_camera()

while not sun_detected and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    filtered_frame = apply_sun_filter(frame)
    top_left, bottom_right = draw_central_box(filtered_frame)
    results = model.infer(filtered_frame)[0]
    detections = sv.Detections.from_inference(results)
    
    if len(detections) > 0:
        sun_detected = True
        for bbox in detections.xyxy:
            sun_center = ((bbox[0] + bbox[2])//2, (bbox[1] + bbox[3])//2)
            dx, dy = calculate_distance_to_box(sun_center, top_left, bottom_right)
            print(f"Initial Sun Detection - Time: {time.strftime('%H:%M:%S')}, dx: {dx}, dy: {dy}")
    else:
        print("Searching for sun...")
    
    annotated_frame = bounding_box_annotator.annotate(scene=filtered_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imshow("Initial Sun Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

turn_off_camera(cap)

if sun_detected:
    print("Sun detected. Starting interval-based tracking.")
    detection_interval = calculate_detection_interval(latitude, longitude, datetime.now(timezone.utc))
    last_detection_time = time.time()

    while True:
        current_time = time.time()
        
        if current_time - last_detection_time >= detection_interval:
            cap = turn_on_camera()
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    filtered_frame = apply_sun_filter(frame)
                    top_left, bottom_right = draw_central_box(filtered_frame)
                    results = model.infer(filtered_frame)[0]
                    detections = sv.Detections.from_inference(results)
                    
                    if len(detections) > 0:
                        for bbox in detections.xyxy:
                            sun_center = ((bbox[0] + bbox[2])//2, (bbox[1] + bbox[3])//2)
                            dx, dy = calculate_distance_to_box(sun_center, top_left, bottom_right)
                            print(f"Time: {time.strftime('%H:%M:%S')}, dx: {dx}, dy: {dy}")
                            
                            detection_interval = calculate_detection_interval(latitude, longitude, datetime.now(timezone.utc))
                    else:
                        print("Sun not detected")
                    
                    annotated_frame = bounding_box_annotator.annotate(scene=filtered_frame, detections=detections)
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
                    cv2.imshow("Filtered Detection", annotated_frame)
                
                turn_off_camera(cap)
            
            last_detection_time = current_time
        
        time.sleep(1)  # Sleep for 1 second to reduce CPU usage
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()