"""
Testing Mode Window - Full display for accuracy testing WITHOUT action execution.
"""

import cv2
import numpy as np

from src.core.gesture_detector import GestureDetector
from src.core.gesture_recognizer import GestureRecognizer
from src.utils.fps_counter import FPSCounter
from src.utils.normalize import normalize_landmarks
from src.ui import ui_utils


class TestingModeWindow:
    """
    Testing mode interface - full display without action execution.

    Purpose: Test and validate gesture recognition accuracy
    Features:
    - Large 1280x720 window
    - Full webcam feed with hand landmarks
    - Gesture name, confidence, FPS display
    - Scale diagnostics
    - Smoothing toggle
    - NO action execution
    """

    def __init__(self, config):
        """
        Initialize testing mode window.

        Args:
            config: AppConfig instance
        """
        self.config = config

        # Initialize components
        self.detector = GestureDetector(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )

        self.recognizer = GestureRecognizer(
            model_path=config.MODEL_PATH,
            gesture_map_path=config.GESTURE_MAP_PATH,
            history_size=config.SMOOTHING_HISTORY_SIZE
        )

        self.fps_counter = FPSCounter()

        # Camera setup
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        # State
        self.current_gesture = "No hand detected"
        self.current_confidence = 0.0
        self.smoothing_enabled = True
        self.last_normalization_result = None

        # Window name
        self.window_name = "Testing Mode - Gesture Recognition (No Actions)"

    def run(self):
        """Main loop for testing mode."""
        print("\n" + "="*60)
        print("TESTING MODE - Gesture Recognition Accuracy Testing")
        print("="*60)
        print("\nFeatures:")
        print("  • Full display with detailed diagnostics")
        print("  • NO action execution (safe for testing)")
        print("  • Smoothing toggle")
        print("\nControls:")
        print("  S: Toggle temporal smoothing on/off")
        print("  Q or ESC: Exit to menu")
        print("="*60 + "\n")

        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.config.TESTING_MODE_SIZE)

        while True:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame from camera")
                break

            # Flip for selfie view
            frame = cv2.flip(frame, 1)

            # Detect hand
            hand_landmarks = self.detector.detect_hand(frame)

            # Predict gesture
            if hand_landmarks:
                # Get normalization result for diagnostics
                self.last_normalization_result = normalize_landmarks(hand_landmarks, threshold=0.05)

                # Predict gesture
                if self.smoothing_enabled:
                    gesture, confidence, valid = self.recognizer.predict_smooth(
                        hand_landmarks, normalize_landmarks
                    )
                else:
                    gesture, confidence, valid = self.recognizer.predict(
                        hand_landmarks, normalize_landmarks
                    )

                if valid:
                    self.current_gesture = gesture
                    self.current_confidence = confidence
                else:
                    self.current_gesture = gesture  # "Invalid (scale too small)"
                    self.current_confidence = 0.0
            else:
                self.current_gesture = "No hand detected"
                self.current_confidence = 0.0
                self.last_normalization_result = None
                self.recognizer.reset_history()

            # Update FPS
            self.fps_counter.update()

            # Render UI
            self._render_frame(frame, hand_landmarks)

            # Display frame
            cv2.imshow(self.window_name, frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('s'):  # Toggle smoothing
                self.smoothing_enabled = not self.smoothing_enabled
                self.recognizer.reset_history()
                status = "ON" if self.smoothing_enabled else "OFF"
                print(f"Temporal smoothing: {status}")

        # Cleanup
        self.cleanup()

    def _render_frame(self, frame, hand_landmarks):
        """
        Render UI elements on frame.

        Args:
            frame: BGR image to draw on (modified in-place)
            hand_landmarks: MediaPipe hand landmarks or None
        """
        height, width = frame.shape[:2]

        # Draw hand landmarks
        if hand_landmarks:
            self.detector.draw_landmarks(frame, hand_landmarks)

        # Draw semi-transparent panel
        ui_utils.draw_semi_transparent_panel(frame, 180, alpha=0.6)

        # Draw gesture label (large, color-coded)
        ui_utils.draw_gesture_label(
            frame,
            self.current_gesture,
            self.current_confidence,
            (20, 50),
            self.config,
            font_scale=1.2
        )

        # Draw confidence
        if hand_landmarks and "Invalid" not in self.current_gesture and self.current_gesture != "No hand detected":
            ui_utils.draw_confidence(
                frame,
                self.current_confidence,
                (20, 85),
                self.config
            )

        # Draw scale diagnostics
        if hand_landmarks and self.last_normalization_result:
            ui_utils.draw_scale_diagnostics(
                frame,
                self.last_normalization_result,
                (20, 115),
                self.config
            )

            # Draw validation status
            status_text = "Status: Valid" if self.last_normalization_result['valid'] else "Status: SKIPPED (scale too small)"
            status_color = self.config.COLOR_GREEN if self.last_normalization_result['valid'] else self.config.COLOR_RED
            cv2.putText(
                frame,
                status_text,
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2
            )

        # Draw FPS counter
        ui_utils.draw_fps_counter(frame, self.fps_counter.get_fps(), self.config)

        # Draw smoothing status
        smoothing_text = "Smoothing: ON" if self.smoothing_enabled else "Smoothing: OFF"
        smoothing_color = self.config.COLOR_GREEN if self.smoothing_enabled else self.config.COLOR_GRAY
        cv2.putText(
            frame,
            smoothing_text,
            (width - 150, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            smoothing_color,
            1
        )

        # Draw instructions
        instructions = "Press 'S' to toggle smoothing | 'Q' or 'ESC' to quit"
        ui_utils.draw_instructions(frame, instructions, self.config)

    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.detector.cleanup()

        print("\n" + "="*60)
        print("Testing Mode ended. Returning to menu...")
        print("="*60)
