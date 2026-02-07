"""
Gesture Detector - Hand detection and landmark extraction using MediaPipe.
"""

import cv2
import mediapipe as mp
import numpy as np


class GestureDetector:
    """
    Handles hand detection and landmark extraction using MediaPipe Hands.

    Responsibilities:
    - Initialize MediaPipe Hands solution
    - Process video frames to detect hands
    - Extract 21 hand landmarks
    - Draw landmarks on frames for visualization
    """

    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        """
        Initialize the gesture detector with MediaPipe.

        Args:
            min_detection_confidence: Minimum confidence for hand detection (0-1)
            min_tracking_confidence: Minimum confidence for landmark tracking (0-1)
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # Video stream mode
            max_num_hands=1,          # Detect only one hand
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect_hand(self, frame):
        """
        Detect hand in the frame and extract landmarks.

        Args:
            frame: BGR image from webcam (numpy array)

        Returns:
            hand_landmarks: MediaPipe hand landmarks object with 21 points, or None if no hand detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe
        results = self.hands.process(rgb_frame)

        # Extract first hand landmarks (if any detected)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]

        return None

    def draw_landmarks(self, frame, hand_landmarks):
        """
        Draw hand landmarks and connections on the frame.

        Args:
            frame: BGR image to draw on (modified in-place)
            hand_landmarks: MediaPipe hand landmarks object
        """
        if hand_landmarks is None:
            return

        # Draw landmarks with custom styling
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 0),  # Green circles
                thickness=2,
                circle_radius=3
            ),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(255, 255, 255),  # White lines
                thickness=2
            )
        )

    def cleanup(self):
        """Release MediaPipe resources."""
        self.hands.close()
