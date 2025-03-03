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
import tensorflow as tf  # For TensorFlow Lite interpreter

warnings.filterwarnings('ignore', category=UserWarning)

# Environment settings to suppress unnecessary logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ORT_LOGGING_LEVEL'] = '3'

# Load configuration (if needed for other parts of the project)
try:
    with open('appConfig.json') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print("appConfig.json not found. Please make sure the file exists in the same directory as the script.")
except json.JSONDecodeError:
    print("Error decoding appConfig.json. Please make sure it's a valid JSON file.")

# Function to draw a central box on the frame
def draw_central_box(frame, box_size=100):
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    top_left = (center_x - box_size // 2, center_y - box_size // 2)
    bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle
    return top_left, bottom_right

# Function to load and initialize the TensorFlow Lite model
def load_tflite_model(model_path):
    """Load and initialize the TensorFlow Lite model."""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"Error loading TensorFlow Lite model: {e}")
        return None

# Function to run inference on a frame using the TensorFlow Lite model
def run_inference(interpreter, frame):
    """Run inference on a frame using the TensorFlow Lite model."""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Preprocess the frame (resize and normalize)
        input_shape = input_details[0]['shape'][1:3]  # Get input shape (height, width)
        resized_frame = cv2.resize(frame, (input_shape[1], input_shape[0]))
        input_data = np.expand_dims(resized_frame / 255.0, axis=0).astype(np.float32)

        # Set input tensor and invoke the model
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get output tensor (assuming single output)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def main():
    # Load your custom TensorFlow Lite model
    model_path = "models/sun_tracker_v3/sun_tracker_v3_float16.tflite"
    interpreter = load_tflite_model(model_path)

    if not interpreter:
        print("Failed to load the TensorFlow Lite model. Exiting...")
        return

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

    # Create VideoWriter object for saving output video
    output_filename = f"sun_detection_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    detection_interval = 0.1  # Start with a short interval for testing purposes
    last_detection_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        if current_time - last_detection_time >= detection_interval:
            try:
                # Run inference on the current frame using the custom model
                results = run_inference(interpreter, frame)

                if results is not None and len(results) > 0:
                    print(f"Inference results: {results}")

                    # Draw central box for visualization purposes
                    top_left, bottom_right = draw_central_box(frame)

                    # Annotate detections on the frame (if applicable)
                    detections = sv.Detections.from_inference(results)  # Adjust this based on your output format
                    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

                    out.write(annotated_frame)  # Write annotated frame to output video

                    cv2.imshow("Sun Detection", annotated_frame)  # Display annotated frame in popup window

                else:
                    print("No detections made.")
                    out.write(frame)  # Write original frame when no detections are made

                last_detection_time = current_time

            except Exception as e:
                print(f"Error during processing: {e}")
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
