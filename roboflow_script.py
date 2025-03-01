from inference import get_model
import supervision as sv
import cv2
import numpy as np
import time
from datetime import datetime, timezone, timedelta
from suncalc import get_position
import math
import geocoder
import os
import json
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress UserWarnings
warnings.filterwarnings('ignore', category=UserWarning)

# Suppress ONNX Runtime warnings
os.environ['ORT_LOGGING_LEVEL'] = '3'

# Load the API key from appConfig.json
try:
    with open('appConfig.json') as config_file:
        config = json.load(config_file)
    api_key = config.get('ROBOFLOW_API_KEY', '')
except FileNotFoundError:
    print("appConfig.json not found. Please make sure the file exists in the same directory as the script.")
    api_key = ''
except json.JSONDecodeError:
    print("Error decoding appConfig.json. Please make sure it's a valid JSON file.")
    api_key = ''

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
    return img

def calculate_sun_movement(lat, lon, time1, time2):
    pos1 = get_position(time1, lat, lon)
    pos2 = get_position(time2, lat, lon)
    movement = math.sqrt((pos2['azimuth'] - pos1['azimuth'])**2 + (pos2['altitude'] - pos1['altitude'])**2)
    return math.degrees(movement)

def calculate_detection_interval(lat, lon, current_time):
    return 0 
    # - for test purposes
    future_time = current_time + timedelta(minutes=5)
    movement = calculate_sun_movement(lat, lon, current_time, future_time)
    if movement > 1:
        return 60
    elif movement > 0.5:
        return 180
    else:
        return 300

def main():
    model = get_model(model_id="sun-tracking-555mn/4", api_key=api_key)
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    g = geocoder.ip('me')
    latitude, longitude = g.latlng

    print(f"Current location: {g.city}, {g.state}, {g.country}")
    print(f"Latitude: {latitude}, Longitude: {longitude}")

    while True:
        choice = input("Do you want to use webcam or video file? (webcam/video): ").lower()
        if choice in ['webcam', 'video']:
            break
        print("Invalid choice. Please enter 'webcam' or 'video'.")

    if choice == 'webcam':
        cap = cv2.VideoCapture("/dev/video0")
    else:
        while True:
            video_path = input("Enter the path to your video file: ")
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                break
            print("File not found. Please enter a valid path.")

    if not cap.isOpened():
        print("Error opening video source")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter object
    output_filename = f"sun_detection_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    sun_detected = False
    detection_interval = 0.1  # Start with 1 second interval
    last_detection_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        
        if current_time - last_detection_time >= detection_interval:
            filtered_frame = apply_sun_filter(frame)
            top_left, bottom_right = draw_central_box(filtered_frame)
            results = model.infer(filtered_frame)[0]
            detections = sv.Detections.from_inference(results)
            
            if len(detections) > 0:
                sun_detected = True
                for bbox in detections.xyxy:
                    sun_center = ((bbox[0] + bbox[2])//2, (bbox[1] + bbox[3])//2)
                    dx, dy = calculate_distance_to_box(sun_center, top_left, bottom_right)
                    print(f"Time: {time.strftime('%H:%M:%S')}, dx: {dx}, dy: {dy}")
                
                detection_interval = calculate_detection_interval(latitude, longitude, datetime.now(timezone.utc))
            else:
                print("Sun not detected")
            
            annotated_frame = bounding_box_annotator.annotate(scene=filtered_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # Write the frame to the output video
            out.write(annotated_frame)
            
            cv2.imshow("Sun Detection", annotated_frame)
            
            last_detection_time = current_time
        else:
            # Write the original frame when not detecting
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Output video saved as {output_filename}")

if __name__ == "__main__":
    main()
