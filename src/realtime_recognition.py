"""
Real-Time Gesture Recognition with Webcam

This script performs live gesture recognition:
- Captures webcam video
- Detects hand with MediaPipe
- Normalizes landmarks (same pipeline as training)
- Predicts gesture using trained model
- Displays result on video feed
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import json
from pathlib import Path
import time
import pyautogui

from utils.normalize import normalize_landmarks


class GestureActionTrigger:
    """
    Hybrid trigger system: detects when a gesture should trigger an action.
    Uses minimum dwell time to prevent false positives, triggers only once per hold.
    """
    
    def __init__(self, min_dwell_frames=5):
        """
        Initialize action trigger.
        
        Args:
            min_dwell_frames: Minimum frames to hold gesture before triggering (default: 5 ≈ 0.15s at 30fps)
        """
        self.min_dwell_frames = min_dwell_frames
        self.current_gesture = None
        self.frame_count = 0
        self.triggered_this_gesture = False
        
    def update(self, detected_gesture):
        """
        Update trigger state with current gesture.
        
        Args:
            detected_gesture: The gesture detected this frame
            
        Returns:
            str or None: Gesture name if action should trigger, None otherwise
        """
        # Ignore non-actionable states
        if detected_gesture in ["null", "No hand detected", "Invalid (scale too small)"]:
            self.current_gesture = None
            self.frame_count = 0
            self.triggered_this_gesture = False
            return None
        
        # Same gesture as previous frame
        if detected_gesture == self.current_gesture:
            self.frame_count += 1
            
            # Trigger only once when dwell threshold first met
            if self.frame_count == self.min_dwell_frames and not self.triggered_this_gesture:
                self.triggered_this_gesture = True
                return detected_gesture  # TRIGGER ACTION
        else:
            # New gesture detected - reset state
            self.current_gesture = detected_gesture
            self.frame_count = 0
            self.triggered_this_gesture = False
        
        return None


class RealtimeGestureRecognizer:
    def __init__(self, model_path=None):
        """
        Initialize the real-time gesture recognizer.
        
        Args:
            model_path: Path to trained model (uses latest if None)
        """
        self.project_root = Path(__file__).parent.parent
        
        # Load gesture map
        with open(self.project_root / "data" / "gesture_map.json", 'r') as f:
            self.gesture_map = json.load(f)
        
        # Load trained model
        if model_path is None:
            model_path = self.project_root / "models" / "gesture_classifier_latest.pkl"
        
        print(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("✓ Model loaded successfully!")
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Webcam setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # State tracking
        self.current_gesture = "No hand detected"
        self.prediction_confidence = 0.0
        self.fps = 0
        self.last_time = time.time()
        
        # For temporal smoothing (optional - reduces flicker)
        self.prediction_history = []
        self.history_size = 5  # Average over last 5 frames
        
        # Action triggering system
        self.action_trigger = GestureActionTrigger(min_dwell_frames=5)
        self.last_action = None
        self.action_display_frames = 0  # Frames to display action feedback
        
    def predict_gesture(self, hand_landmarks):
        """
        Predict gesture from hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            tuple: (gesture_name, confidence, normalized_features_valid)
        """
        # Normalize landmarks (same as training)
        result = normalize_landmarks(hand_landmarks, threshold=0.05)
        
        if result is None or not result['valid']:
            return "Invalid (scale too small)", 0.0, False
        
        # Get normalized coordinates
        features = result['normalized'].reshape(1, -1)  # Shape: (1, 42)
        
        # Predict
        prediction = self.model.predict(features)[0]
        
        # Get prediction probabilities (if model supports it)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities[prediction]
        else:
            confidence = 1.0  # k-NN doesn't have probabilities
        
        # Map prediction ID to gesture name
        gesture_name = self.gesture_map.get(str(prediction), "Unknown")
        
        return gesture_name, confidence, True
    
    def smooth_prediction(self, gesture_name):
        """
        Apply temporal smoothing to reduce prediction flickering.
        Uses majority voting over the last N frames.
        
        Args:
            gesture_name: Current frame's prediction
            
        Returns:
            str: Smoothed gesture name
        """
        self.prediction_history.append(gesture_name)
        
        # Keep only last N predictions
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Return most common prediction
        if self.prediction_history:
            # Count occurrences
            counts = {}
            for pred in self.prediction_history:
                counts[pred] = counts.get(pred, 0) + 1
            
            # Return most frequent
            return max(counts, key=counts.get)
        
        return gesture_name
    
    def handle_action(self, gesture):
        """
        Execute action based on triggered gesture.
        Clicks at current cursor position (system-level, works across all apps).
        
        Args:
            gesture: Name of the gesture that triggered the action
        """
        if gesture == "pinch":
            # Click at current cursor position
            pyautogui.click()
            self.last_action = "LEFT CLICK"
            self.action_display_frames = 30  # Show feedback for ~1 second
            print(f"✓ LEFT CLICK executed (pinch gesture)")
        
        # Add more gesture → action mappings here:
        # elif gesture == "fist":
        #     pyautogui.click(button='right')
        #     self.last_action = "RIGHT CLICK"
        # elif gesture == "thumbs_up":
        #     pyautogui.scroll(100)  # Scroll up
        #     self.last_action = "SCROLL UP"
    
    def draw_ui(self, frame, hand_landmarks):
        """
        Draw UI overlay on frame showing prediction and info.
        
        Args:
            frame: Video frame
            hand_landmarks: MediaPipe hand landmarks (or None)
        """
        height, width, _ = frame.shape
        
        # Draw hand landmarks if detected
        if hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
        
        # Create semi-transparent overlay panel
        overlay = frame.copy()
        panel_height = 120
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Display current gesture (large text)
        gesture_text = f"Current Gesture: {self.current_gesture}"
        
        # Choose color based on gesture
        if self.current_gesture == "No hand detected":
            text_color = (128, 128, 128)  # Gray
        elif "Invalid" in self.current_gesture:
            text_color = (0, 0, 255)  # Red
        elif self.current_gesture == "null":
            text_color = (200, 200, 200)  # Light gray
        else:
            text_color = (0, 255, 0)  # Green (active gesture)
        
        cv2.putText(
            frame,
            gesture_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            text_color,
            3
        )
        
        # Display confidence if available
        if hand_landmarks and "Invalid" not in self.current_gesture and self.current_gesture != "No hand detected":
            confidence_text = f"Confidence: {self.prediction_confidence*100:.1f}%"
            cv2.putText(
                frame,
                confidence_text,
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Display action feedback (when action was just triggered)
        if self.action_display_frames > 0:
            action_text = f"ACTION: {self.last_action}"
            # Pulsing effect based on remaining frames
            alpha = min(1.0, self.action_display_frames / 15.0)
            color_intensity = int(255 * alpha)
            cv2.putText(
                frame,
                action_text,
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, color_intensity, 0),  # Green, fading
                2
            )
            self.action_display_frames -= 1
        
        # Display FPS in top-right corner
        fps_text = f"FPS: {self.fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(
            frame,
            fps_text,
            (width - text_size[0] - 20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        
        # Instructions at bottom
        instructions = "Press 'Q' or 'ESC' to quit | 'S' to toggle smoothing | PINCH = Click"
        cv2.putText(
            frame,
            instructions,
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        
        return frame
    
    def calculate_fps(self):
        """Calculate and update FPS."""
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
    
    def run(self):
        """Main loop for real-time gesture recognition."""
        print("\n" + "="*60)
        print("REAL-TIME GESTURE RECOGNITION")
        print("="*60)
        print("\nControls:")
        print("  Q or ESC: Quit")
        print("  S: Toggle temporal smoothing on/off")
        print("\nStarting webcam...")
        print("="*60 + "\n")
        
        smoothing_enabled = True
        
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Get hand landmarks
            hand_landmarks = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
            
            # Predict gesture
            if hand_landmarks:
                gesture, confidence, valid = self.predict_gesture(hand_landmarks)
                
                if valid:
                    # Apply smoothing if enabled
                    if smoothing_enabled:
                        gesture = self.smooth_prediction(gesture)
                    
                    self.current_gesture = gesture
                    self.prediction_confidence = confidence
                    
                    # Check if gesture should trigger an action
                    triggered = self.action_trigger.update(gesture)
                    if triggered:
                        self.handle_action(triggered)
                else:
                    self.current_gesture = gesture  # "Invalid (scale too small)"
                    self.prediction_confidence = 0.0
            else:
                self.current_gesture = "No hand detected"
                self.prediction_confidence = 0.0
                self.prediction_history = []  # Clear history when hand disappears
                self.action_trigger.update("No hand detected")  # Reset trigger state
            
            # Calculate FPS
            self.calculate_fps()
            
            # Draw UI
            frame = self.draw_ui(frame, hand_landmarks)
            
            # Show smoothing status
            if smoothing_enabled:
                cv2.putText(
                    frame,
                    "Smoothing: ON",
                    (frame.shape[1] - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
            
            # Display frame
            cv2.imshow('Real-Time Gesture Recognition', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('s'):  # Toggle smoothing
                smoothing_enabled = not smoothing_enabled
                self.prediction_history = []  # Clear history when toggling
                status = "ON" if smoothing_enabled else "OFF"
                print(f"Temporal smoothing: {status}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("Session ended. Goodbye!")
        print("="*60)


def main():
    """Entry point."""
    recognizer = RealtimeGestureRecognizer()
    recognizer.run()


if __name__ == "__main__":
    main()
