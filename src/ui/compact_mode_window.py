"""
Compact Mode Window - Small always-on-top window WITH action execution.
"""

import cv2
import numpy as np

from src.core.gesture_detector import GestureDetector
from src.core.gesture_recognizer import GestureRecognizer
from src.core.action_executor import ActionExecutor
from src.utils.fps_counter import FPSCounter
from src.utils.normalize import normalize_landmarks
from src.ui import ui_utils


class CompactModeWindow:
    """
    Compact mode interface - small always-on-top window with action execution.

    Purpose: Unobtrusive gesture control during regular computer use
    Features:
    - Small 320x240 window (resizable)
    - Always-on-top behavior
    - Moveable and minimizable
    - Small webcam preview with hand landmarks
    - Minimal UI: gesture + confidence only
    - EXECUTES ACTIONS (e.g., pinch → click)
    """

    def __init__(self, config):
        """
        Initialize compact mode window.

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

        self.action_executor = ActionExecutor(
            gesture_actions=config.GESTURE_ACTIONS,
            min_dwell_frames=config.ACTION_DWELL_FRAMES
        )

        self.fps_counter = FPSCounter()

        # Camera setup
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        # State
        self.current_gesture = "No hand detected"
        self.current_confidence = 0.0

        # Window name
        self.window_name = "Compact Mode - Gesture Control"

    def run(self):
        """Main loop for compact mode."""
        print("\n" + "="*60)
        print("COMPACT MODE - Gesture Control")
        print("="*60)
        print("\nFeatures:")
        print("  • Small, always-on-top window")
        print("  • Gesture recognition with action execution")
        print("  • Moveable and minimizable")
        print("\nActive gesture mappings:")
        for gesture, action in self.config.GESTURE_ACTIONS.items():
            print(f"  {gesture} → {action}")
        print("\nControls:")
        print("  Q or ESC: Exit to menu")
        print("="*60 + "\n")

        # Create resizable window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.window_name, *self.config.COMPACT_MODE_SIZE)

        # Set always-on-top
        try:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            print("✓ Always-on-top enabled")
        except:
            print("⚠ Warning: Always-on-top not supported on this platform")

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
                gesture, confidence, valid = self.recognizer.predict_smooth(
                    hand_landmarks, normalize_landmarks
                )

                if valid:
                    self.current_gesture = gesture
                    self.current_confidence = confidence

                    # Update action trigger
                    triggered_gesture = self.action_executor.update(gesture)

                    # Execute action if triggered
                    if triggered_gesture:
                        self.action_executor.execute_action(triggered_gesture)
                else:
                    self.current_gesture = gesture  # "Invalid (scale too small)"
                    self.current_confidence = 0.0
                    self.action_executor.reset()
            else:
                self.current_gesture = "No hand detected"
                self.current_confidence = 0.0
                self.recognizer.reset_history()
                self.action_executor.reset()

            # Update FPS
            self.fps_counter.update()

            # Render UI
            self._render_frame(frame, hand_landmarks)

            # Check if window is minimized
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)

            # Display frame (only if window is visible)
            if visible > 0:
                cv2.imshow(self.window_name, frame)
            # Note: Detection and action execution continue even when minimized!

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                break

        # Cleanup
        self.cleanup()

    def _render_frame(self, frame, hand_landmarks):
        """
        Render minimal UI elements on frame.

        Args:
            frame: BGR image to draw on (modified in-place)
            hand_landmarks: MediaPipe hand landmarks or None
        """
        height, width = frame.shape[:2]

        # Draw hand landmarks (compact version)
        if hand_landmarks:
            self.detector.draw_landmarks(frame, hand_landmarks)

        # Draw semi-transparent panel (smaller than testing mode)
        ui_utils.draw_semi_transparent_panel(frame, 80, alpha=0.7)

        # Draw gesture label (compact, smaller font)
        color = ui_utils.get_gesture_color(self.current_gesture, self.config)
        gesture_text = f"{self.current_gesture}"

        cv2.putText(
            frame,
            gesture_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # Draw confidence (compact)
        if hand_landmarks and "Invalid" not in self.current_gesture and self.current_gesture != "No hand detected":
            conf_text = f"{self.current_confidence*100:.1f}%"
            cv2.putText(
                frame,
                conf_text,
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.config.COLOR_WHITE,
                1
            )

        # Draw action feedback (compact)
        if self.action_executor.action_display_frames > 0:
            action_text = self.action_executor.last_action

            # Pulsing effect
            alpha = min(1.0, self.action_executor.action_display_frames / 15.0)
            color_intensity = int(255 * alpha)

            cv2.putText(
                frame,
                action_text,
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, color_intensity, 0),  # Green, fading
                2
            )

            self.action_executor.decrement_display_frames()

    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.detector.cleanup()

        print("\n" + "="*60)
        print("Compact Mode ended. Returning to menu...")
        print("="*60)
