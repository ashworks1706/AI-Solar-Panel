# AI Solar Panel: Advanced Sun Tracking System

An innovative self-adjusting solar panel system that maximizes energy absorption using computer vision and deep learning.

## Project Overview

This repository contains the full implementation of our AI-powered solar panel tracking system. The project optimizes solar energy capture by accurately tracking the sun's position in real-time, using a lightweight deep learning model running on a Raspberry Pi with motor control via microcontroller.


![image](https://github.com/user-attachments/assets/eea5721c-b19e-4e94-939d-d3f80633a671)

![image](https://github.com/user-attachments/assets/fc0dd9a4-0482-4a62-8def-da6433f52e5f)

## Technical Architecture

### Computer Vision Pipeline

The system employs a sophisticated computer vision pipeline built with OpenCV and Ultralytics YOLOv8:

1. **Image Acquisition**: Captures frames from a camera module at dynamically calculated intervals
2. **Sun Detection**: Processes frames through our custom-trained YOLOv8 model
3. **Position Analysis**: Calculates the sun's offset from center using pixel coordinates
4. **Motor Control Commands**: Translates positional data into servo motor instructions

```python
def calculate_distance(center_x, center_y, bbox):
    """Calculates the distance of the detected object from the center."""
    x1, y1, x2, y2 = bbox
    object_center_x = (x1 + x2) / 2
    object_center_y = (y1 + y2) / 2
    distance_x = object_center_x - center_x
    distance_y = object_center_y - center_y
    return distance_x, distance_y
```

![image](https://github.com/user-attachments/assets/cb524227-639e-42ce-a87a-55f1a9b45cd9)



![image](https://github.com/user-attachments/assets/a70c042f-330c-4c49-9386-214e1cefd40a)

### Weather-Adaptive Scheduling

The system intelligently manages detection frequency based on weather conditions:

- **Clear Sky**: 60-second intervals for optimal tracking
- **Partly Cloudy**: 180-second intervals to balance accuracy and power usage
- **Overcast/Rainy**: 300-second intervals to conserve power during low solar output
- **Nighttime**: Extended sleep mode until sunrise to maximize energy efficiency

```python
def calculate_next_interval():
    """Calculate the next interval time based on weather conditions and time of day"""
    # Check if it's nighttime (after sunset or before sunrise)
    is_nighttime = current_time > sunset or current_time < sunrise
    
    if is_nighttime:
        # If sun is set, use a very long interval until next sunrise
        time_until_sunrise = sunrise - current_time if current_time < sunrise else (sunrise + 86400) - current_time
        new_interval = min(int(time_until_sunrise), 3600)  # Cap at 1 hour max
    else:
        # Base interval on weather conditions during daytime
        if "clear" in weather_condition or cloud_coverage < 20:
            new_interval = 60  # 1 minute
        elif "cloud" in weather_condition or cloud_coverage < 70:
            new_interval = 180  # 3 minutes
        else:
            new_interval = 300  # 5 minutes
```


## Key Features

- **Real-time Sun Position Detection**: Utilizes a custom-trained YOLOv8 model to detect the sun's position with 98.8% mAP50-95 accuracy
- **Intelligent Weather Adaptation**: Dynamically adjusts detection intervals based on cloud coverage and time of day
- **Day/Night Cycle Awareness**: Conserves power by entering low-power mode during nighttime hours
- **Comprehensive Monitoring Dashboard**: Next.js web interface for real-time system monitoring and control
- **Robust Data Logging**: Firebase integration for performance tracking and system diagnostics
- **Automated Fallback Mechanisms**: Implements non-ML algorithms for tracking during adverse weather conditions
- **Optimized Power Management**: Camera activates only when needed to conserve energy

### Real-time Monitoring Dashboard

The Next.js dashboard provides comprehensive system monitoring and control:

- **System Status Overview**: Camera state, interval settings, and weather conditions
- **Sun Position Visualization**: Real-time graphical representation of detected sun position
- **Performance Metrics**: CPU, memory, and disk usage monitoring
- **Weather Data Display**: Current conditions with sunrise/sunset visualization
- **Control Panel**: Camera activation, interval adjustment, and test mode controls


![image](https://github.com/user-attachments/assets/47335071-2c20-493f-9474-72f4e5de00c5)

![image](https://github.com/user-attachments/assets/263f00e9-31bd-4c2e-92aa-bcf440ed7cfc)

![image](https://github.com/user-attachments/assets/010c34b7-d4cd-489e-84f1-952a0bc1208f)

![image](https://github.com/user-attachments/assets/5658303b-1b3e-42e5-9a68-147aa8356478)


![image](https://github.com/user-attachments/assets/251e4fba-2b79-47d3-964e-e6c5e91af843)

![image](https://github.com/user-attachments/assets/908d1270-42eb-47bd-b705-28c26f4ddce1)



### Data Logging and Analysis

The system uses Firebase Firestore for comprehensive data logging:

- **ModelLog Collection**: Stores detection results, system performance metrics, and timestamps
- **ProgramLog Collection**: Records weather data, interval calculations, and system events
- **Test Mode Data**: Captures continuous testing data for system optimization

## Technologies Used

### Backend Technologies

- **Flask**: Powers the RESTful API server with endpoints for system control and monitoring
- **OpenCV**: Handles image processing and camera operations with efficient frame manipulation
- **Ultralytics YOLOv8**: Provides the core object detection capabilities with state-of-the-art accuracy
- **Supervision**: Simplifies detection visualization and post-processing
- **Firebase Admin SDK**: Enables secure database operations for logging and monitoring
- **Flask-CORS**: Ensures secure cross-origin resource sharing for the web dashboard
- **python-dotenv**: Manages environment variables for secure API key storage

### Frontend Technologies

- **Next.js**: Creates a responsive, server-rendered React application
- **TailwindCSS**: Provides utility-first CSS for rapid UI development
- **NextUI**: Delivers modern UI components with accessibility features
- **Firebase SDK**: Enables real-time data synchronization with the backend


## Model Performance

Our custom-trained YOLOv8 model achieves exceptional performance for sun detection:

| Model                             | mAP50-95 | Precision | Recall | Images | Type         | Suitability for Microcontroller |
| --------------------------------- | -------- | --------- | ------ | ------ | ------------ | ------------------------------- |
| **Our YOLO model v3**       | 98.8%    | 91.7%     | 92.4%  | 3409   | YOLOv8       | ✅ Excellent (Lightweight)      |
| sun-tracking-555mn/4              | 97.7%    | 93.0%     | 89.2%  | 923    | YOLOv8n      | ✅ Good (Roboflow Library)      |
| solar-re1fe/1                     | 96.8%    | 95.2%     | 94.3%  | 2684   | Roboflow 3.0 | ✅ Good (Optimized for Edge)    |
| sun-tracking-photovoltaic-panel/1 | 98.2%    | 93.7%     | 93.7%  | 196    | Roboflow 2.0 | ⚠️ Limited Dataset            |
| sun-tracking/3                    | 92.5%    | 94.7%     | 91.8%  | 1090   | YOLOv5       | ✅ Compatible (Proven on MCUs)  |
| Our YOLO model v2                 | 42.6%    | 78.7%     | 79.3%  | 274    | YOLOv8       | ❌ Too Heavy                    |
| Our YOLO model v1                 | 21.3%    | 58.0%     | 64.0%  | 198    | YOLOv8       | ❌ Not Viable                   |



The model was trained on a comprehensive dataset combining:
- Sun Tracking Photovoltaic Panel Dataset
- Sun Dataset
- Sun Tracking Dataset
- Solar Dataset
- Sun Detection Dataset
- Custom dataset with 400 manually labeled images

## API Endpoints

The Flask server exposes several RESTful endpoints:

- **PUT /start_stop_camera**: Toggles camera and model detection on/off
- **PUT /change_interval**: Updates the detection interval timing
- **POST /test_model**: Activates continuous testing mode for system validation
- **GET /status**: Returns comprehensive system status information

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
   ```

5. Set up Firebase
   - Get firebase-secret.json from your Firebase project
   - Place it in your root directory

6. Start the Flask server
   ```bash
   python main.py
   ```

7. Run the dashboard (in a separate terminal)
   ```bash
   cd ..
   npm install
   npm run dev
   ```

8. Configure hardware components
   - Connect Raspberry Pi to camera module
   - Set up the microcontroller for motor control
   - Ensure proper power connections for all components

The dashboard will be available at http://localhost:3000, and the Flask API will be running at http://localhost:5000.

## System Testing

The system includes a dedicated test mode that:
- Continuously captures and processes frames
- Logs detection results to Firebase
- Monitors system performance metrics
- Validates weather data integration
- Tests interval calculation algorithms

This comprehensive testing approach ensures reliable operation in various conditions.

## Future Enhancements

- Integration with machine learning for predictive weather analysis
- Power consumption optimization through advanced sleep modes
- Enhanced motor control algorithms for smoother tracking
- Mobile application for remote monitoring and control
- Integration with smart home systems via MQTT

## Contributors

- **@Ampers8nd (Justin Erd.)**: Mechanical design
- **@Zhoujjh3 (Justin Zhou)**: Electrical components
- **@ashworks1706 (Ash S.)**: Deep learning development

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.