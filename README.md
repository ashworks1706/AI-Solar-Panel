# AI Solar Panel

An innovative self-adjusting solar panel system that maximizes energy absorption using computer vision and deep learning.

## Project Overview

This repository contains the deep learning component of our AI-powered solar panel tracking system. The project aims to optimize solar energy capture by accurately tracking the sun's position in real-time, using a lightweight deep learning model running on a microcontroller.

![image](https://github.com/user-attachments/assets/eea5721c-b19e-4e94-939d-d3f80633a671)

![image](https://github.com/user-attachments/assets/fc0dd9a4-0482-4a62-8def-da6433f52e5f)

![image](https://github.com/user-attachments/assets/cb524227-639e-42ce-a87a-55f1a9b45cd9)

![image](https://github.com/user-attachments/assets/5658303b-1b3e-42e5-9a68-147aa8356478)

![image](https://github.com/user-attachments/assets/47335071-2c20-493f-9474-72f4e5de00c5)

![image](https://github.com/user-attachments/assets/263f00e9-31bd-4c2e-92aa-bcf440ed7cfc)

![image](https://github.com/user-attachments/assets/4c98cd9c-7e5d-4294-bbe0-3f8c235ed5f1)


## Key Features

- **Sun Position Detection**: Utilizes a lightweight deep learning model for real-time sun tracking in various weather conditions.
- **Microcontroller Integration**: Designed to run efficiently on a Raspberry Pi with Teensy/Arduino for motor control.
- **Automated Adjustment**: Provides coordinates to guide the solar panel's orientation for maximum sunlight exposure.
- **Fallback Algorithm**: Implements a non-ML algorithm for tracking during heavy cloud cover or rain.
- **Performance Tracking**: Monitors wattage generation and usage for comparison between algorithmic and deep-learning modes.
- **Adaptive Detection Intervals**: Adjusts detection frequency based on weather conditions to optimize power usage.
- **Camera Management**: Activates the camera only when needed for detection to conserve energy.
- **Weather Integration**: Uses weather API data to intelligently schedule detection intervals.
- **Web Dashboard**: Provides monitoring and control through a Next.js admin interface

| Model                             | mAP50-95 | Precision | Recall | Images | Type         | Suitability for Microcontroller |
| --------------------------------- | -------- | --------- | ------ | ------ | ------------ | ------------------------------- |
| **Our YOLO model v3**       | 98.8%    | 91.7%     | 92.4%  | 3409   | YOLOv8       | ✅ Excellent (Lightweight)      |
| sun-tracking-555mn/4              | 97.7%    | 93.0%     | 89.2%  | 923    | YOLOv8n      | ✅ Good (Roboflow Library)      |
| solar-re1fe/1                     | 96.8%    | 95.2%     | 94.3%  | 2684   | Roboflow 3.0 | ✅ Good (Optimized for Edge)    |
| sun-tracking-photovoltaic-panel/1 | 98.2%    | 93.7%     | 93.7%  | 196    | Roboflow 2.0 | ⚠️ Limited Dataset            |
| sun-tracking/3                    | 92.5%    | 94.7%     | 91.8%  | 1090   | YOLOv5       | ✅ Compatible (Proven on MCUs)  |
| Our YOLO model v2                 | 42.6%    | 78.7%     | 79.3%  | 274    | YOLOv8       | ❌ Too Heavy                    |
| Our YOLO model v1                 | 21.3%    | 58.0%     | 64.0%  | 198    | YOLOv8       | ❌ Not Viable                   |

1. **Our YOLO model v3**

   - **Best Choice for Deployment**
   - 98.8% mAP50-95 (Custom Trained)
   - 55.5ms inference on CPU
   - Needs:
   - Current size: 87MB → 20mb after quantization
2. **sun-tracking-555mn/4**

   - **Custom Solution Potential**
   - 97.7% mAP at 93.0ms inference
   - YOLOv8n architecture (4.9MB) fits Teensy 4.1's 1MB Flash
   - Confirmed working with LibRT for ARM Cortex-M7
3. **solar-re1fe/1**

   - **Backup Option**
   - Roboflow's optimized TF-Lite model
   - Requires 64KB RAM (fits most MCUs)
   - Pre-trained weights reduce development time
4. **sun-tracking/3**

   - **Baseline Reference**
   - YOLOv5s (14.0MB) requires STM32H7 series
   - Proven C++ implementation exists
5. **Legacy Models (v1/v2)**

   - **Development History**
   - Demonstrates iterative improvement
   - Not suitable for deployment

## System Architecture

1. **Hardware Layer**:

   - 480p-720p camera captures sky images
   - Raspberry Pi hosts the detection model and Flask server
   - Microcontroller controls servo motors based on detection results
   - Servo motors adjust panel's azimuth and attitude
2. **Software Layer**:

   - **Flask Server**: Central control system that manages:
     - Detection scheduling based on weather conditions
     - Communication between components
     - Logging and monitoring
   - **Deep Learning Model**: YOLO-based model detects sun position
   - **C++ Motor Control**: Script that receives coordinates from Flask server
   - **Next.js Dashboard**: Admin interface for monitoring and control

## Technical Details

- **Model**: Lightweight YOLOv8 model for sun detection
- **Datasets Used for Training**:

  1. **Sun Tracking Photovoltaic Panel Dataset**

  - Workspace: yassine-pzpt7
  - Project: sun-tracking-photovoltaic-panel
  - Version: 2

  2. **Sun Dataset**

  - Workspace: 1009727588-qq-com
  - Project: sun-nxvfz
  - Version: 2

  3. **Sun Tracking Dataset**

  - Workspace: rik-tjduw
  - Project: sun-tracking-555mn
  - Version: 4

  4. **Solar Dataset**

  - Workspace: stardetect
  - Project: solar-re1fe
  - Version: 1

  5. **Sun Detection Dataset**

  - Workspace: fruitdetection-ulcz9
  - Project: sun_detection-hl04q
  - Version: 1

  6. **Custom Dataset**
     - 400 Images from google of clear sky and sun
     - Custom labelling using labellmg
- **Server Framework**: Flask for API endpoints and system management
- **Hardware**:

  - Raspberry Pi for model hosting and server
  - Teensy or Arduino for motor control
- **Additional Sensors**: Inertial Measurement Unit (IMU) for current position and orientation
- **Programming Languages**:

  - Python for Flask server and deep learning
  - Next.JS for web dashboard and firebase
  - C++ for firmware and motor control mechanisms
- **APIs**:

  - Weather API for cloud coverage and sun visibility prediction
  - Firebase for logging and debugging
- **Power Management**: Ensures peak wattage draw does not exceed average power generated

## Flask Server Endpoints

The system relies on a Flask server with the following endpoints:

- **PUT Endpoints**:

  - `/start_stop_camera`: Toggles camera and model detection
  - `/change_interval`: Updates the detection interval timing
- **POST Endpoints**:

  - `/getmodelresponse`: Processes images or video and returns predictions
  - `/postcurrentstatus`: Logs model and Raspberry Pi status to Firebase
  - `/postprogramdetails`: Logs program details including weather data
- **GET Endpoints**:

  - `/status`: Returns current system status including camera state and intervals

## **ModelLog Collection**:

In the project, Firebase Firestore is used as the primary database for logging system operations and model detections. The Flask server interacts with two main Firestore collections to store critical data:

- Stores detection results from the YOLO model
- Captures system performance metrics like CPU and memory usage
- Contains timestamps of each detection event
- When Firebase Storage is enabled, includes links to captured images
- Example document structure:
  ```
  {
    "model_details": {
      "detections": [
        {
          "bbox": [240.5, 180.3, 320.7, 260.8],
          "confidence": 0.92,
          "class_id": 0,
          "distance_x": 20.5,
          "distance_y": -15.2
        }
      ],
      "timestamp": "2025-03-08T16:47:23.456789"
    },
    "raspberry_details": {
      "cpu_percent": 35.2,
      "memory_percent": 42.8,
      "disk_percent": 56.7
    },
    "image_url": "https://storage.example.com/solar_panel_images/20250308_164723.jpg",
    "timestamp": "2025-03-08T16:47:23.456789"
  }
  ```

2. **ProgramLog Collection**:
   - Records weather API responses
   - Stores interval calculation formulas and logic
   - Tracks next scheduled detection times
   - Documents manual interval changes from the dashboard
   - Example document structure:
     ```
     {
       "weather_response": {
         "weather_condition": "Partly Cloudy",
         "weather_description": "scattered clouds",
         "temperature": 293.15,
         "clouds": 45,
         "wind_speed": 3.2,
         "timestamp": "2025-03-08T16:47:23.456789"
       },
       "interval_formula": "Based on Partly Cloudy with 45% cloud coverage",
       "next_interval_time": 1741050443.456789,
       "timestamp": "2025-03-08T16:47:23.456789"
     }
     ```

These Firestore collections are queried by the Next.js dashboard to display real-time system status and historical data. The Flask server's endpoints `/postcurrentstatus` and `/postprogramdetails` handle writing to these collections, while the frontend periodically retrieves and displays this data to provide administrators with comprehensive monitoring capabilities[1][2].

This database structure enables efficient debugging, performance analysis, and historical tracking of the solar panel's operation over time, forming a critical part of the system's monitoring infrastructure.

## Mechanical Design

- 3D printed components for rapid prototyping
- Designed to fit within a specified volume (TBD)
- Incorporates waterproofing and stability features
- Accommodates all necessary equipment (PCBs, camera, servos, gears, photovoltaic cell)

## Electrical Design

- Utilizes two PWM servos for azimuth and attitude control
- Incorporates appropriate energy storage solution (battery)
- Designed to operate within the power constraints of the solar panel

## Computer Vision Implementation

The system uses advanced computer vision techniques for sun tracking:

- Pixel grid mapping with labeled intersections for precise positioning
- Central crosshair reference point for calculating sun offset
- Distance calculation from center to detected sun position
- Support for both image and video processing

## [Model Tests are here](https://github.com/ashworks1706/AI-Solar-Panel/tree/main/models)

#### Latest sun_tracker_v3 model demo

https://github.com/user-attachments/assets/0c48888f-dec7-4363-8610-fab38e45028e

## Setup and Installation

1. Clone the repository

   ```bash
   git clone https://github.com/ashworks1706/AI-Solar-Panel.git
   cd AI-Solar-Panel
   ```
2. Create a virtual environment with Python 3.11

   ```bash
   cd python
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies

   ```bash
   pip install -r python/requirements.txt
   ```
4. Configure environment variables for weather API

   - Create a `.env` file in the project root with your API keys:

   ```
   WEATHER_API_KEY=your_openweather_api_key
   WEATHER_LAT=37.7749
   WEATHER_LON=-122.4194
   ```

   and get firebase-secret.json from your project, place it in your root directory.
5. Start the Flask server

   ```bash
   python main.py
   ```
6. Run the dashboard (in a separate terminal)

   ```bash
   cd ..
   npm install
   npm run dev
   ```
7. Configure hardware components

   - Connect Raspberry Pi to camera module
   - Set up the microcontroller for motor control
   - Ensure proper power connections for all components

The dashboard will be available at http://localhost:3000, and the Flask API will be running at http://localhost:5000.

## Current Progress

### Mechanical

- [X] Select appropriate solar panel based on wattage requirements
- [X] Select appropriate energy storage solution (battery)
- [X] Create initial design sketches
- [ ] Develop preliminary mechanical design
- [ ] Conduct initial FEA analysis for wind effects and stability
- [ ] Implement waterproofing measures

### Software / Vision

- [X] Implement YOLO model for sun detection
- [X] Develop adaptive detection interval algorithm
- [X] Implement camera management for power conservation
- [X] Create fallback algorithm for adverse weather conditions
- [X] Finalize model selection (sun_tracker_v3)
- [X] Implement Flask server for system control
- [X] Integrate weather API for intelligent interval scheduling
- [X] Develop Firebase logging for debugging and monitoring
- [X] Create Next.js admin dashboard interface
- [X] Complete integration with C++ motor control system

## Contributors

- **@Ampers8nd (Justin Erd.)**: Mechanical design
- **@Zhoujjh3 (Justin Zhou)**: Electrical components
- **@ashworks1706 (Ash S.)**: Deep learning development

## Contributing

We welcome contributions to improve any aspect of our solar panel system. Please open an issue or submit a pull request with your suggestions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
