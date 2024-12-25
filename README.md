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

## System Architecture

1. 480p-720p camera captures sky images
2. Deep learning model detects sun's position
3. Microcontroller processes input and calculates optimal panel alignment
4. Servo motors adjust panel's azimuth and attitude

## Technical Details

- **Model**: Lightweight deep learning model (potentially YOLOv5)
- **Hardware**: Teensy or Arduino Microcontroller
- **Additional Sensors**: Inertial Measurement Unit (IMU) for current position and orientation
- **Programming Languages**: C++ for firmware and control mechanisms, potentially Python for the deep learning model
- **Power Management**: Ensures peak wattage draw does not exceed average power generated

## Mechanical Design

- 3D printed components for rapid prototyping
- Designed to fit within a specified volume (TBD)
- Incorporates waterproofing and stability features
- Accommodates all necessary equipment (PCBs, camera, servos, gears, photovoltaic cell)

## Electrical Design

- Utilizes two PWM servos for azimuth and attitude control
- Incorporates appropriate energy storage solution (battery)
- Designed to operate within the power constraints of the solar panel

## Setup and Installation

1. Clone the repository
2. Install dependencies (list to be added)
3. Configure hardware components
4. Run the main program

Detailed instructions will be provided in the future.

## Current Progress

### Mechanical
- [ ] Select appropriate solar panel based on wattage requirements
- [ ] Select appropriate energy storage solution (battery)
- [ ] Create initial design sketches
- [ ] Develop preliminary mechanical design
- [ ] Conduct initial FEA analysis for wind effects and stability
- [ ] Implement waterproofing measures

### Software / Vision
- [ ] Research lightweight models for microcontrollers (YOLOv5, YOLOv8)
- [ ] Select optimal dataset for training
- [ ] Train and evaluate models
- [ ] Implement and test fallback algorithm

## Future Improvements

- Optimize deep learning model for microcontroller deployment
- Enhance fallback algorithm for adverse weather conditions
- Improve software-hardware integration

## Contributors

- **@Ampers8nd (Justin Erd.)**: Mechanical design
- **@Zhoujjh3 (Justin Zhou)**: Electrical components
- **@somwrks (Ash S.)**: Deep learning development

## Contributing

We welcome contributions to improve any aspect of our solar panel system. Please open an issue or submit a pull request with your suggestions.

## License

This project is licensed under the MIT License - see the [MIT](LICENSE) file for details.
