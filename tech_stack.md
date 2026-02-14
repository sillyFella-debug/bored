# Tech Stack for "Boredom" Heatmap System

## Core Technology Stack

### Programming Language
- **Python** - Industry standard for computer vision applications

### Screen Capture
- **mss** - Ultra-fast cross-platform screen capture library
- Alternative: pyautogui, PIL (Pillow)

### Computer Vision
- **MediaPipe** - Face Mesh solution for real-time face tracking
- **OpenCV (cv2)** - Image processing and manipulation
- **NumPy** - Numerical computations and array handling

### Audio Processing
- **pygame** - Play audio files (air horn sound)
- Alternative: playsound, pyaudio

### Audio Routing Solution
- **VB-Audio Virtual Cable** - Route system audio to Zoom microphone input
- Alternative: Jack Audio Connection Kit (Linux), Soundflower (macOS - deprecated)

## Core Libraries Installation
```bash
pip install opencv-python mediapipe mss numpy pygame
```

## Key Technical Components

### 1. Screen Capture Module
- Region of Interest (ROI) definition
- Continuous screen capture at ~10 FPS
- Cropping to Zoom gallery view area

### 2. Face Detection & Analysis
- MediaPipe Face Mesh for facial landmarks
- Head pose estimation (yaw/pitch detection)
- Mouth aspect ratio (MAR) calculation for yawn detection

### 3. Boredom Detection Algorithm
- Calculate percentage of participants looking away
- Detect yawning based on mouth openness duration (>2.5 seconds)
- Combine metrics into "Bored Score"

### 4. Audio Alert System
- Air horn sound file playback
- Audio routing through virtual cable to avoid echo cancellation

## Implementation Challenges

### 1. Resolution Issues
- Faces in Zoom gallery are small (~100x100 pixels)
- Solution: Focus on macro movements (head pose, mouth opening) instead of micro-expressions

### 2. False Positive Reduction
- Differentiate between talking and yawning
- Implement time buffer for yawn detection (minimum 2.5 seconds)

### 3. Audio Delivery
- Bypass Zoom's noise suppression
- Use virtual audio cable to route sound directly to microphone input

## Additional Requirements

### Hardware
- Webcam-enabled computer
- Stable internet connection for Zoom
- Audio output device

### Software Environment
- Python 3.7+ environment
- Zoom client installed
- Administrative privileges for installing virtual audio cables