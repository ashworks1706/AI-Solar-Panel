# AI Solar Panel

An innovative self-adjusting solar panel system that maximizes energy absorption using computer vision and deep learning.

## Project Overview

This repository contains the deep learning component of our AI-powered solar panel tracking system. The project aims to optimize solar energy capture by accurately tracking the sun's position in real-time, using a lightweight deep learning model running on a microcontroller.

## Key Features

- **Sun Position Detection**: Utilizes a lightweight deep learning model for real-time sun tracking in various weather conditions.
- **Microcontroller Integration**: Designed to run efficiently on a Teensy or Arduino microcontroller.
- **Automated Adjustment**: Provides coordinates to guide the solar panel's orientation for maximum sunlight exposure.
- **Fallback Algorithm**: Implements a non-ML algorithm for tracking during heavy cloud cover or rain.
- **Performance Tracking**: Monitors wattage generation and usage for comparison between algorithmic and deep-learning modes.
- **Adaptive Detection Intervals**: Adjusts detection frequency based on sun movement to optimize power usage.
- **Camera Management**: Activates the camera only when needed for detection to conserve energy.

## System Architecture

1. 480p-720p camera captures sky images
2. Deep learning model detects sun's position
3. Microcontroller processes input and calculates optimal panel alignment
4. Servo motors adjust panel's azimuth and attitude

## Technical Details

- **Model**: Lightweight deep learning model (YOLOv8 converted via LiteRT)
- **Hardware**: Teensy or Arduino Microcontroller
- **Additional Sensors**: Inertial Measurement Unit (IMU) for current position and orientation
- **Programming Languages**: C++ for firmware and control mechanisms, Python for the deep learning model
- **Power Management**: Ensures peak wattage draw does not exceed average power generated
- **Location Detection**: Automatically determines the system's geographical location for accurate sun tracking

## Mechanical Design

- 3D printed components for rapid prototyping
- Designed to fit within a specified volume (TBD)
- Incorporates waterproofing and stability features
- Accommodates all necessary equipment (PCBs, camera, servos, gears, photovoltaic cell)

## Electrical Design

- Utilizes two PWM servos for azimuth and attitude control
- Incorporates appropriate energy storage solution (battery)
- Designed to operate within the power constraints of the solar panel


## [Tests are here](https://github.com/ashworks1706/AI-Solar-Panel/tree/main/models)

#### Latest sun_tracker_v3 model demo

https://github.com/user-attachments/assets/0c48888f-dec7-4363-8610-fab38e45028e

## Setup and Installation

1. Clone the repository
2. Install dependencies
3. Configure hardware components
4. Run the main program

## Current Progress

### Mechanical

- [ ] Select appropriate solar panel based on wattage requirements
- [ ] Select appropriate energy storage solution (battery)
- [ ] Create initial design sketches
- [ ] Develop preliminary mechanical design
- [ ] Conduct initial FEA analysis for wind effects and stability
- [ ] Implement waterproofing measures

### Software / Vision

- [X] Implement YOLOv8 model for sun detection
- [X] Develop adaptive detection interval algorithm
- [X] Implement camera management for power conservation
- [X] Create fallback algorithm for adverse weather conditions
- [ ] Finalize which model to use whether custom or Roboflow
- [ ] Integrate automatic location detection
- [ ] Optimize model for microcontroller deployment
- [ ] Conduct comprehensive testing and performance analysis

## Contributors

- **@Ampers8nd (Justin Erd.)**: Mechanical design
- **@Zhoujjh3 (Justin Zhou)**: Electrical components
- **@somwrks (Ash S.)**: Deep learning development

## Contributing

We welcome contributions to improve any aspect of our solar panel system. Please open an issue or submit a pull request with your suggestions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
