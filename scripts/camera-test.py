import cv2

def test_video_capture_and_save():
    """Test the webcam by displaying a live feed and saving it to a file."""
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

# Run the function
if __name__ == "__main__":
    test_video_capture_and_save()
