# from inference import get_model
import supervision as sv
import cv2
import numpy as np
import time
from datetime import datetime, timezone, timedelta
from suncalc import get_position
import geocoder
import os
import json
import warnings
import math

warnings.filterwarnings('ignore', category=UserWarning)

# os.environ['QT_QPA_PLATFORM'] = 'wayland'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ORT_LOGGING_LEVEL'] = '3'

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

def test_camera():
    """Test the camera by recording a 5-second video, displaying it in a popup, and drawing a central box."""
    try:
        # Open the default camera (index 0)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Unable to access the camera.")
            return

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 FPS if unavailable

        # Define the codec and create VideoWriter object
        output_filename = "webcam_recording.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

        print("Recording... Press 'q' to stop.")

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from the camera.")
                break

            # Write the frame to the output file
            out.write(frame)

            # Display the resulting frame
            cv2.imshow("Webcam Feed - Recording", frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopping recording...")
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Release resources in all cases
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals():
            out.release()
        cv2.destroyAllWindows()

        print(f"Recording saved as {output_filename}" if os.path.exists(output_filename) else "Recording failed.")

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
        cap = cv2.VideoCapture(0)
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
                    sun_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    dx, dy = calculate_distance_to_box(sun_center, top_left, bottom_right)
                    print(f"Time: {time.strftime('%H:%M:%S')}, dx: {dx}, dy: {dy}")

                detection_interval = calculate_detection_interval(latitude, longitude,
                                                                  datetime.now(timezone.utc))
            else:
                print("Sun not detected")

            annotated_frame = bounding_box_annotator.annotate(scene=filtered_frame,
                                                              detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame,
                                                       detections=detections)

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
    while True:
        initial_choice = input("Do you want to test the camera? (yes/no): ").lower()
        if initial_choice in ['yes', 'no']:
            break
        print("Invalid input. Please enter 'yes' or 'no'.")

    if initial_choice == 'yes':
        print("Testing the camera...")
        test_camera()
    
    main()