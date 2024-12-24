# AI Solar Panel

This repository contains the deep learning component of an innovative self-adjusting solar panel system. Our project aims to maximize solar energy absorption by accurately tracking the sun's position using computer vision and deep learning techniques, running on a microcontroller.

## Key Features

- **Sun Position Detection**: Utilizes a lightweight deep learning model for real-time sun tracking in various weather conditions.
- **Microcontroller Integration**: Designed to run efficiently on a Teensy or Arduino microcontroller.
- **Automated Adjustment**: Provides coordinates to guide the solar panel's orientation for maximum sunlight exposure.
- **Fallback Algorithm**: Implements a non-ML algorithm for tracking during heavy cloud cover or rain.
- **Performance Tracking**: Monitors wattage generation and usage for comparison between algorithmic and deep-learning modes.

## System Architecture

The system operates as follows:
1. A 480p-720p camera captures images of the sky
2. The deep learning model detects the sun's position
3. The microcontroller processes the input and calculates the optimal panel alignment
4. Servo motors adjust the panel's azimuth and attitude based on the calculated coordinates

## Technical Details

- **Model**: Lightweight deep learning model (potentially YOLOv5)
- **Hardware**: Teensy or Arduino Microcontroller
- **Additional Sensors**: Inertial Measurement Unit (IMU) for current position and orientation
- **Programming Languages**: C for firmware and control mechanisms, potentially Python for the deep learning model
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

1. Install all requirements (specific instructions to be added)
2. Run the main Python file
3. Ensure all components are properly connected and calibrated

## Contributors

- **@Ampers8nd (Justin Erd.)**: Handled mechanical design
- **@Zhoujjh3 (Justin Zhou)**: Managed electrical components
- **@somwrks (Ash S.)**: Developed deep learning component

## Future Improvements

- Optimize the deep learning model for microcontroller deployment
- Enhance the fallback algorithm for better performance in adverse weather conditions
- Improve integration between software and hardware components

## Contributing

We welcome contributions to improve any aspect of our solar panel system. Please refer to our contributing guidelines for more information.
