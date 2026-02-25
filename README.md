# Student Attention Monitoring System

## Overview

The **Student Attention Monitoring System** is a computer-vision based application that helps track and analyze a student's attentiveness during online or offline learning sessions.  
Using a webcam feed, the system detects the student’s face and eyes, estimates head pose, and infers whether the student is focused on the screen or distracted.

This project is intended for educational and research purposes, to demonstrate how classical computer vision and basic machine learning techniques can be combined to build an attention‑tracking tool.

## Key Features

- **Real‑time face detection**: Uses OpenCV (and optionally pre‑trained models) to detect faces from the webcam stream.
- **Eye / gaze proxy**: Tracks eye region and head orientation as a proxy for attention.
- **Attention state classification**: Labels frames as “Attentive” or “Not Attentive” based on defined thresholds (e.g., looking away, eyes closed for a period of time).
- **Session statistics**: Calculates overall attention percentage for a session.
- **Configurable thresholds**: You can tune sensitivity (e.g., how long looking away counts as distraction).

> Note: The exact feature set may differ slightly depending on how you extend or customize the project.

## Tech Stack

- **Language**: Python 3.x  
- **Core libraries**:
  - OpenCV (`cv2`) for video capture and image processing
  - NumPy for numerical operations
  - (Optional) dlib / mediapipe / other models for more robust face and landmark detection

## Project Structure (typical)

- `main.py` – entry point for running the attention monitoring application  
- `models/` – any trained models or cascades (e.g., Haar cascades)  
- `utils/` – helper functions for preprocessing, drawing overlays, etc.  
- `reports/` – generated attention reports or logs (ignored by Git via `.gitignore`)  

Your actual structure might vary, but this gives a general idea.

## Getting Started

### Prerequisites

- Python 3.8+ installed
- A working webcam
- Recommended: a virtual environment (`venv`)

### Installation

```bash
git clone https://github.com/aksaqi313/Student-Attention-Monitoring-System-.git
cd "Student-Attention-Monitoring-System-"

# (optional) create and activate virtual env
python -m venv .venv
.\.venv\Scripts\activate   # on Windows

pip install -r requirements.txt
```

> If you don’t have a `requirements.txt` yet, install at least:
> - `opencv-python`
> - `numpy`
> and any other libraries you use in `main.py`.

### Running the Application

```bash
python main.py
```

Common behavior:

- A window opens showing the webcam feed.
- Overlays (boxes/text) indicate whether the student is currently **Attentive** or **Not Attentive**.
- At the end of a session, the script can print a summary attention score or save a simple report.

## Configuration

Inside `main.py` (or a config file), you can usually adjust:

- Attention threshold (e.g., minimum percentage of time looking at screen)
- Time window for detecting drowsiness / eye closure
- Camera index (default is `0`)

Adjust these according to your environment and camera position.

## Limitations & Ethics

- The system is **not** a perfect measure of learning or understanding—only a rough estimate of visual attention.
- Lighting conditions, camera quality, and occlusions (e.g., glasses, masks) can reduce accuracy.
- Always inform users that they are being recorded/monitored and obtain proper consent.
- Use this project responsibly and in accordance with privacy and data‑protection laws.

## License

This project is licensed under the **Apache License 2.0**.  
See the `LICENSE` file in this repository for the full license text.

In short, you are free to:

- Use the code commercially or privately
- Modify and distribute it
- Sublicense it

As long as you:

- Include a copy of the Apache-2.0 license
- State significant changes you make
- Provide required notices and attributions

For the full legal terms, please refer to the official text in `LICENSE` or at  
`https://www.apache.org/licenses/LICENSE-2.0`.

## Acknowledgements

- The Python `.gitignore` template is adapted from the official GitHub gitignore templates.
- OpenCV and the broader open‑source community for providing the tools that make this project possible.