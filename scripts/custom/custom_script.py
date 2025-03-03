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
from supervision import Detections
import tensorflow as tf  # For TensorFlow Lite interpreter

warnings.filterwarnings('ignore', category=UserWarning)

# Environment settings to suppress unnecessary logs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only mode for TensorFlow Lite

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
    """Load and initialize TFLite model with error handling"""
    try:
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=4  # Optimize for multi-core CPUs
        )
        interpreter.allocate_tensors()
        print(f"Model {model_path} loaded successfully")
        return interpreter
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


# Function to run inference on a frame using the TensorFlow Lite model
def run_inference(interpreter, frame):
    """Run inference with proper preprocessing for 640-trained model"""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Get model input specifications
        input_shape = input_details[0]['shape'][1:3]  # Expected (640, 640)
        dtype = input_details[0]['dtype']

        # Maintain aspect ratio with letterbox resize
        height, width = frame.shape[:2]
        scale = min(input_shape[0]/height, input_shape[1]/width)
        new_size = (int(width*scale), int(height*scale))
        
        # Resize with antialiasing and preserve aspect ratio
        resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
        
        # Create canvas with model's expected size
        canvas = np.full((input_shape[0], input_shape[1], 3), 114, dtype=np.uint8)
        canvas[0:new_size[1], 0:new_size[0]] = resized
        
        # Convert to float32 and normalize (same as training)
        input_data = np.expand_dims(canvas, axis=0).astype(np.float32) / 255.0

        # Set tensor and invoke
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        return interpreter.get_tensor(output_details[0]['index']).squeeze()
    
    except Exception as e:
        print(f"Inference error: {e}")
        return None


# If you want labels, add class_id to your Detections
def parse_detections(output_data):
    """Parse detections with placeholder class_id"""
    try:
        xyxy = []
        confidence = []
        class_id = []
        
        for detection in output_data:
            if detection[-1] > 0.5:
                xyxy.append(detection[:4])
                confidence.append(detection[-1])
                class_id.append(0)  # Single class (sun) = class 0
        
        return Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            class_id=np.array(class_id)
        )
    except Exception as e:
        print(f"Error parsing detections: {e}")
        return Detections.empty()


def main():
    # Load your custom TensorFlow Lite model
    model_path = "models/sun_tracker_v3/sun_tracker_v3_float16.tflite"
    interpreter = load_tflite_model(model_path)

    if not interpreter:
        print("Failed to load the TensorFlow Lite model. Exiting...")
        return

    bounding_box_annotator = sv.BoxAnnotator(
    color_lookup=sv.ColorLookup.INDEX  # Use detection index for coloring
)
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

                if results is not None:
                    detections = parse_detections(results)

                    if len(detections) > 0:
                        print(f"Detections: {detections}")

                        # Draw central box for visualization purposes
                        top_left, bottom_right = draw_central_box(frame)

                        # Annotate detections on the frame (if applicable)
                        annotated_frame = bounding_box_annotator.annotate(
                            scene=frame,
                            detections=detections
                        )
                        # Add label annotations if needed
                        class_names = {0: "sun", 1:"class_0"}
                        annotated_frame = label_annotator.annotate(
                            scene=annotated_frame,
                            detections=detections,
                            labels=[class_names[cid] for cid in detections.class_id]
                        )


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
