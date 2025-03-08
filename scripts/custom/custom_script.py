import cv2
import time
from datetime import datetime
import os
import json
import warnings
import math
from supervision import Detections
from ultralytics import YOLO 

warnings.filterwarnings('ignore', category=UserWarning)

# Environment settings to suppress unnecessary logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# Load configuration (if needed for other parts of the project)
try:
    with open('appConfig.json') as config_file:
        config = json.load(config_file)
except (FileNotFoundError, NameError):
    print("Configuration file not found or JSON module not imported. Continuing without config.")

# Class names - simplified for YOLO
class_names = {0: "sun"}  # Primary class for sun tracking

print("Class names loaded:")
print(class_names)

# Function to draw a central box on the frame
def draw_central_box(frame, box_size=50):
    """Draws a central box on the frame."""
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    top_left = (center_x - box_size // 2, center_y - box_size // 2)
    bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle for crosshair
    return center_x, center_y
# Function to load YOLO model
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
# Convert YOLO results to Supervision Detections format
def yolo_to_detections(yolo_result):
    """Convert YOLO results to Supervision Detections format"""
    try:
        if yolo_result is None or len(yolo_result.boxes) == 0:
            return Detections.empty()
            
        # Extract boxes, confidence scores, and class IDs
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

# Process single image with YOLO
def test_image(image_path, model, class_names):
    """Process single image with YOLO"""
    try:
        # Validate image path
        if not os.path.isfile(image_path):
            print(f"Error: Image file not found: {image_path}")
            return
            
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error loading image: {image_path}")
            return
            
        # Create output directory
        os.makedirs("results", exist_ok=True)
        
        # Process image with YOLO
        results = model.predict(source=frame, conf=0.3, verbose=False)[0]
        if results is None:
            print("Inference returned no results")
            cv2.imshow("Test Result (No detections)", frame)
            cv2.waitKey(0)
            return

        # Use YOLO's built-in visualization
        annotated_frame = results.plot()
        
        # For compatibility with existing code, convert to supervision format
        detections = yolo_to_detections(results)

        print("Detection results:")
        print(detections)
        
        # Save and display
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"results/test_output_{timestamp}.jpg"
        cv2.imwrite(output_path, annotated_frame)
        print(f"Saved to: {output_path}")
        
        cv2.imshow("Test Result", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Image processing failed: {e}")

# Process video with YOLO
def test_video(video_path, model, class_names):
    """Process video file with YOLO"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw central grid box
        center_x, center_y = draw_central_box(frame)
        
        # Perform YOLO inference
        results = model.predict(source=frame, conf=0.3)[0]
        
        if results.boxes:
            for box in results.boxes.xyxy.cpu().numpy():
                # Draw bounding box around detected sun
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue bounding box
                
                # Calculate and display distances from the center
                distance_x, distance_y = calculate_distance(center_x, center_y, (x1, y1, x2, y2))
                cv2.putText(frame,
                            f"dx: {distance_x:.1f}, dy: {distance_y:.1f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1)
        
        # Display frame with annotations
        cv2.imshow("Processing Video", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Process webcam with YOLO
def run_webcam(model, class_names):
    """Process webcam feed with YOLO"""
    try:
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error opening webcam")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output directory
        os.makedirs("results", exist_ok=True)
        
        # Setup video writer
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"results/webcam_output_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
        
        print("Processing webcam feed...")
        print(f"Press 'q' to quit, 's' to save a snapshot")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading from webcam")
                break
                
            try:
                # Run inference with YOLO
                results = model.predict(source=frame, conf=0.3, verbose=False)[0]
                
                if results is not None:
                    # Use YOLO's built-in visualization
                    annotated_frame = results.plot()
                    
                    # Add frame counter and time
                    cv2.putText(
                        annotated_frame,
                        f"Frame: {frame_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    # Write and display
                    out.write(annotated_frame)
                    cv2.imshow("Webcam Processing", annotated_frame)
                    
                else:
                    # On inference failure, show original frame
                    cv2.putText(
                        frame,
                        "Inference failed",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                    out.write(frame)
                    cv2.imshow("Webcam Processing", frame)
            
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                out.write(frame)
                cv2.imshow("Webcam Processing", frame)
                
            frame_count += 1
            
            # Check for keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Webcam processing stopped by user")
                break
            elif key == ord('s'):
                # Save snapshot
                snapshot_path = f"results/snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(snapshot_path, annotated_frame if 'annotated_frame' in locals() else frame)
                print(f"Snapshot saved to: {snapshot_path}")
                
        # Cleanup and report
        elapsed_time = time.time() - start_time
        fps_processing = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"Webcam processing complete!")
        print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
        print(f"Average processing speed: {fps_processing:.2f} FPS")
        
    except Exception as e:
        print(f"Webcam processing failed: {e}")
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        if 'out' in locals() and out is not None:
            out.release()
        cv2.destroyAllWindows()

def main():
    try:
        # Model selection with validation
        model_path = input("Enter model path (or press Enter for default): ").strip()
        if not model_path:
            model_path = "/run/media/ash/OS/Users/Som/Desktop/AI-Solar-Panel/models/sun_tracker_v3/sun_tracker_v3_float32.tflite"  # Updated default path for YOLO model
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return
            
        # Load YOLO model
        model = load_yolo_model(model_path)
        if not model:
            print("Failed to load the model. Exiting...")
            return

        # User interface with error handling
        while True:
            choice = input("Do you want to use webcam, video, or test an image? (webcam/video/image): ").lower()
            if choice in ['webcam', 'video', 'image']:
                break
            print("Invalid choice. Please enter 'webcam', 'video', or 'image'.")
            
        # Process based on user choice
        if choice == 'image':
            while True:
                image_path = input("Enter image file path: ")
                if os.path.isfile(image_path):
                    test_image(image_path, model, class_names)
                    break
                print("Invalid image path. Please try again.")
                
        elif choice == 'video':
            video_path = input("Enter the path to your video file: ")
            if os.path.exists(video_path):
                test_video(video_path, model, class_names)
            else:
                print("Invalid video path.")
                
        elif choice == 'webcam':
            run_webcam(model, class_names)
                
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
