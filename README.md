# Smart Kitchen

Smart Kitchen is a Human-Computer Interaction project that combines a physical kitchen setup with a digital cooking assistant.

The project uses TUIO markers, Bluetooth, gesture recognition, gaze tracking, computer vision, and an AR component to allow users to interact with the system in different ways.

## About the Project

The main idea is to make the cooking process more interactive.

Instead of using only a keyboard and mouse, the user can place physical markers representing ingredients and kitchen tools. The system detects these markers and uses their position and rotation to control different parts of the application.

The system can also identify different users and load their preferences, which allows the cooking suggestions and interface to change depending on the active profile.

## Main Features

- TUIO-based tangible interaction
- Physical ingredient and kitchen-tool markers
- Marker rotation for quantity interaction
- Personalized user profiles
- Bluetooth-based user identification
- Recipe and meal suggestions
- Calorie and health information
- Gesture recognition
- Gaze tracking
- Circular menu
- Contextual CRUD operations
- Computer vision and object detection
- Unity-based AR functionality
- Chef and client profiles

## TUIO Interaction

TUIO is used to detect and track physical fiducial markers.

The C# application receives TUIO events and uses the marker position, rotation, and session information to control the application.

The project includes the TUIO C# library and the related demo applications.

The main application is implemented in `TuioDemo.cs`.

## Kitchen Markers

The following marker IDs are used by the Smart Kitchen application:

| Marker ID | Object |
|-----------:|--------|
| 0 | Chicken |
| 1 | Tomato |
| 2 | Onion |
| 3 | Spices |
| 4 | Rice |
| 5 | Oil |
| 6 | Spoon |
| 7 | Pot |
| 8 | Knife |

The rotation of some markers can also be used as an input for changing quantities.

## User Profiles

The application supports user profiles containing information such as:

- Favorite meals
- Favorite recipes
- Disliked ingredients
- Blocked ingredient markers
- Accent theme
- Welcome message
- User role
- Cooking preferences

The profile data used in the public repository is demo data.

The profile file is:

`SmartKitchenProfiles.txt`

## Bluetooth

Bluetooth is used as one of the ways to identify the active user.

After a user is identified, the application can load the corresponding profile and use the stored preferences when displaying meals and recipes.

Bluetooth functionality is implemented through the Python components included in the project.

## Gesture Recognition

The project includes a Python-based gesture recognition component.

Gestures can be used as an additional way of interacting with the Smart Kitchen application without relying only on physical markers or a mouse.

The gesture component communicates with the main application using a local socket connection.

## Gaze Tracking

Gaze tracking is another input method included in the project.

The gaze component communicates with the C# application through a local socket and can be used to track where the user is looking during interaction.

## Circular Menu

Marker `20` is used for the circular menu.

The menu provides access to different functions of the Smart Kitchen application through tangible interaction.

## Login and Profiles

Marker `25` can be used for login/profile interaction.

The system can load a user's profile and use the stored information to personalize the interface and cooking suggestions.

## CRUD Operations

The project includes contextual CRUD functionality.

Different marker IDs can be used for operations related to creating, reading, updating, and deleting information within the application.

The CRUD markers are:

- Marker 11
- Marker 12
- Marker 13
- Marker 14

## Computer Vision

The project also contains computer-vision components written in Python.

A YOLO model is included in the project and is used by the camera/object-detection part of the system.

The computer-vision components are kept separate from the main C# application and communicate with it when required.

## Augmented Reality

The project includes a Unity-based AR component.

Marker `30` is used to trigger the AR functionality.

The AR part is included as an additional way of presenting information during the cooking interaction.

## Heatmap

Marker `50` is associated with the heatmap functionality.

The heatmap can be used to visualize gaze or interaction-related information collected by the system.

## Communication

Some of the system components communicate through local sockets.

| Component | Port |
|-----------|------|
| Gesture communication | 5000 |
| Gaze communication | 5001 |

The Python components and the C# application use these connections to exchange information during runtime.

## Project Structure

The main files and folders include:

```text
smart-kitchen/
│
├── TUIO/
├── OSC.NET/
├── PyBluez-0.23/
├── people/
│
├── SmartKitchenProfiles.txt
├── TUIO_CSHARP.sln
│
├── TUIO_DEMO.csproj
├── TUIO_DUMP.csproj
├── TUIO_LIB.csproj
│
├── TuioDemo.cs
├── TuioDemoObject.cs
├── TuioDump.cs
│
├── gesture_control.py
├── Bleak Bluetooth.py
├── bluetooth_profile_scanner_socket_professional.py
│
├── app.config
├── LICENSE.txt
├── README.md
└── README.txt
