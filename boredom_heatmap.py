"""
Boredom Heatmap - Meeting Engagement Tracker

This script monitors Zoom meeting participants' engagement via webcam/screen capture.
If more than 50% of participants are looking away or yawning, it plays an air horn sound.
"""

import cv2
import mediapipe as mp
import mss
import numpy as np
import pygame
import time
from typing import List, Tuple, Optional


class BoredomHeatmap:
    def __init__(self, roi: Tuple[int, int, int, int], boredom_threshold: float = 0.5):
        """
        Initialize the Boredom Heatmap system
        
        Args:
            roi: Region of interest (x, y, width, height) for screen capture
            boredom_threshold: Percentage threshold to trigger air horn (default 0.5 = 50%)
        """
        self.roi = {
            "top": roi[1],
            "left": roi[0],
            "width": roi[2],
            "height": roi[3]
        }
        self.boredom_threshold = boredom_threshold
        
        # Initialize MediaPipe components
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Face mesh configuration
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=20,  # Support up to 20 faces in gallery view
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize pygame for audio
        pygame.mixer.init()
        
        # Load air horn sound (you'll need to provide this file)
        try:
            pygame.mixer.music.load("air_horn.mp3")
        except:
            print("Warning: air_horn.mp3 not found. Using beep sound instead.")
            # Create a simple beep if no air horn file exists
            self.create_beep_sound()
        
        # Store face tracking data for yawn detection
        self.face_yawn_timers = {}
        self.yawn_duration_threshold = 2.5  # seconds to consider a yawn
        
        # Timing for frame capture
        self.last_capture_time = 0
        self.capture_interval = 0.1  # 10 FPS

    def create_beep_sound(self):
        """Create a simple beep sound if air horn file is missing"""
        import wave
        import struct
        
        sample_rate = 44100
        duration = 1  # seconds
        frequency = 1000  # Hz
        
        # Generate beep
        samples = []
        for i in range(int(sample_rate * duration)):
            value = int(32767.0 * 0.5 * np.sin(2.0 * np.pi * frequency * i / sample_rate))
            samples.append([value, value])  # stereo
        
        # Write to temporary file
        with wave.open("temp_beep.wav", "w") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b''.join(struct.pack('<hh', d[0], d[1]) for d in samples))
        
        pygame.mixer.music.load("temp_beep.wav")

    def calculate_mouth_aspect_ratio(self, landmarks, image_shape) -> float:
        """
        Calculate the mouth aspect ratio to detect if mouth is open
        
        Args:
            landmarks: MediaPipe face landmarks
            image_shape: Shape of the image (height, width)
            
        Returns:
            Mouth aspect ratio (higher values indicate more open mouth)
        """
        # Mouth landmark indices (based on MediaPipe face mesh)
        # Inner mouth: 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
        inner_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        
        # Get coordinates for vertical mouth measurements
        # Top inner lip: 78, Bottom inner lip: 308
        # Left inner corner: 291, Right inner corner: 61
        try:
            top_lip = landmarks.landmark[78]
            bottom_lip = landmarks.landmark[308]
            left_corner = landmarks.landmark[291]
            right_corner = landmarks[61]
            
            # Convert normalized coordinates to pixel coordinates
            h, w = image_shape[:2]
            top_y = int(top_lip.y * h)
            bottom_y = int(bottom_lip.y * h)
            left_x = int(left_corner.x * w)
            right_x = int(right_corner.x * w)
            
            # Calculate distances
            vertical_dist = abs(top_y - bottom_y)
            horizontal_dist = abs(right_x - left_x)
            
            # Calculate mouth aspect ratio
            if horizontal_dist == 0:
                return 0.0
            mar = vertical_dist / horizontal_dist
            
            return mar
        except:
            return 0.0

    def is_looking_away(self, landmarks, image_shape) -> bool:
        """
        Determine if person is looking away from camera based on head pose
        
        Args:
            landmarks: MediaPipe face landmarks
            image_shape: Shape of the image (height, width)
            
        Returns:
            Boolean indicating if person is looking away
        """
        # Get nose tip and eye positions to estimate head orientation
        try:
            nose_tip = landmarks.landmark[1]
            left_eye = landmarks.landmark[159]  # Left eye lower eyelid
            right_eye = landmarks.landmark[386]  # Right eye lower eyelid
            
            h, w = image_shape[:2]
            
            # Convert to pixel coordinates
            nose_x = nose_tip.x * w
            left_eye_x = left_eye.x * w
            right_eye_x = right_eye.x * w
            
            # Estimate if nose is significantly off-center relative to eyes
            # This is a simplified approach - a full head pose estimation would be more accurate
            center_of_eyes = (left_eye_x + right_eye_x) / 2
            distance_from_center = abs(nose_x - center_of_eyes)
            
            # If nose is too far from center of eyes, likely looking away
            # Adjust threshold based on testing
            eye_distance = abs(right_eye_x - left_eye_x)
            threshold = eye_distance * 0.5  # Adjust this multiplier as needed
            
            return distance_from_center > threshold
        except:
            return False

    def detect_boredom_in_frame(self, frame) -> Tuple[int, int]:
        """
        Analyze a frame to count looking away and yawning faces
        
        Args:
            frame: Image frame to analyze
            
        Returns:
            Tuple of (count_looking_away, count_yawning)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        looking_away_count = 0
        yawning_count = 0
        
        if results.multi_face_landmarks:
            current_time = time.time()
            
            for idx, landmarks in enumerate(results.multi_face_landmarks):
                # Check if looking away
                if self.is_looking_away(landmarks, frame.shape):
                    looking_away_count += 1
                
                # Check for yawning
                mar = self.calculate_mouth_aspect_ratio(landmarks, frame.shape)
                
                # Define a threshold for what constitutes a yawn
                # This threshold may need adjustment based on testing
                yawn_threshold = 0.5
                
                if mar > yawn_threshold:
                    # Person is currently showing mouth open behavior
                    if idx not in self.face_yawn_timers:
                        self.face_yawn_timers[idx] = current_time
                    elif current_time - self.face_yawn_timers[idx] >= self.yawn_duration_threshold:
                        # Has been consistently "yawning" for the required duration
                        yawning_count += 1
                else:
                    # Reset timer if mouth is closed
                    if idx in self.face_yawn_timers:
                        del self.face_yawn_timers[idx]
        
        return looking_away_count, yawning_count

    def play_air_horn(self):
        """Play the air horn sound"""
        print("BOREDOM DETECTED! Playing air horn...")
        pygame.mixer.music.play()

    def run(self):
        """
        Main execution loop for the boredom heatmap
        """
        print("Starting Boredom Heatmap Monitor...")
        print(f"ROI: {self.roi}")
        print(f"Boredom threshold: {self.boredom_threshold * 100}%")
        print("Press Ctrl+C to stop")
        
        with mss.mss() as sct:
            try:
                while True:
                    current_time = time.time()
                    
                    # Limit capture rate to reduce CPU usage
                    if current_time - self.last_capture_time >= self.capture_interval:
                        # Capture screen region
                        screenshot = sct.grab(self.roi)
                        
                        # Convert to numpy array (BGRA format)
                        frame = np.array(screenshot)
                        
                        # Convert BGRA to BGR (remove alpha channel)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        
                        # Analyze frame for boredom indicators
                        looking_away_count, yawning_count = self.detect_boredom_in_frame(frame)
                        
                        total_faces = len(self.face_yawn_timers.keys()) if self.face_yawn_timers else max(
                            looking_away_count + yawning_count, 1
                        )
                        
                        # Calculate boredom score
                        if total_faces > 0:
                            boredom_score = (looking_away_count + yawning_count) / total_faces
                            
                            print(f"Faces: {total_faces}, Looking away: {looking_away_count}, Yawning: {yawning_count}, Boredom Score: {boredom_score:.2f}")
                            
                            # Trigger air horn if boredom threshold is exceeded
                            if boredom_score > self.boredom_threshold:
                                self.play_air_horn()
                        
                        self.last_capture_time = current_time
                    
                    # Small delay to prevent excessive CPU usage
                    time.sleep(0.01)
                    
            except KeyboardInterrupt:
                print("\nStopping Boredom Heatmap Monitor...")
                
        # Clean up
        self.face_mesh.close()


def main():
    """
    Example usage of the Boredom Heatmap system
    
    You'll need to adjust the ROI (region of interest) to match your Zoom gallery view
    The ROI format is (x, y, width, height) in screen coordinates
    """
    # Example ROI - adjust these values to match your Zoom gallery view
    # Format: (x, y, width, height) - adjust based on your screen setup
    zoom_gallery_roi = (100, 100, 800, 600)  # Example values - customize for your setup
    
    boredom_detector = BoredomHeatmap(
        roi=zoom_gallery_roi,
        boredom_threshold=0.5  # Trigger when 50% or more appear bored
    )
    
    boredom_detector.run()


if __name__ == "__main__":
    main()