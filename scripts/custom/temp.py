from ultralytics import YOLO
import cv2
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Load the YOLO model
def load_yolo_model(model_path):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

# Run inference on an image
def test_image_with_yolo(image_path, model):
    try:
        if not os.path.isfile(image_path):
            print(f"Error: Image file not found: {image_path}")
            return
        
        # Load the image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error loading image: {image_path}")
            return
        
        # Run inference
        results = model.predict(source=frame, save=False, conf=0.3)
        
        # Annotate results on the image
        annotated_frame = results[0].plot()
        
        # Save and display results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"results/test_output_{timestamp}.jpg"
        os.makedirs("results", exist_ok=True)
        cv2.imwrite(output_path, annotated_frame)
        
        # Convert BGR to RGB for display with matplotlib
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(annotated_frame_rgb)
        plt.axis('off')
        plt.title("Detection Results")
        plt.show()
        
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error during image inference: {e}")

# Main function to handle user input
def main():
    try:
        # Load the YOLO model
        model_path = input("Enter YOLO model path (or press Enter for default): ").strip()
        if not model_path:
            model_path = "/run/media/ash/OS/Users/Som/Desktop/AI-Solar-Panel/models/sun_tracker_v3/sun_tracker_v3_float32.tflite"  # Replace with your default model path
        
        yolo_model = load_yolo_model(model_path)
        if not yolo_model:
            print("Failed to load YOLO model. Exiting...")
            return
        
        # User input for mode selection
        while True:
            choice = input("Do you want to use webcam, video, or test an image? (webcam/video/image): ").lower()
            if choice in ['webcam', 'video', 'image']:
                break
            print("Invalid choice. Please enter 'webcam', 'video', or 'image'.")
        
        if choice == 'image':
            while True:
                image_path = input("Enter image file path: ")
                if os.path.isfile(image_path):
                    test_image_with_yolo(image_path, yolo_model)
                    break
                print("Invalid image path. Please try again.")
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
