"""
Compact Mode Window - Small always-on-top window WITH action execution.
"""
import subprocess
import time
import cv2
import numpy as np
import pyautogui

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

        self.settings_button_rect = (50, 50, 20, 20)
        self.settings_button_hovered = False

        self.recognizer = GestureRecognizer(
            model_path=config.MODEL_PATH,
            gesture_map_path=config.GESTURE_MAP_PATH,
            history_size=config.SMOOTHING_HISTORY_SIZE
        )

        self.action_executor = ActionExecutor(
            gesture_actions=config.GESTURE_ACTIONS,
            min_dwell_frames=config.ACTION_DWELL_FRAMES
        )

        # Configure volume control parameters from config
        self.action_executor.volume_increment_interval = config.VOLUME_INCREMENT_INTERVAL
        self.action_executor.volume_increment_percent = config.VOLUME_INCREMENT_PERCENT
        self.action_executor.volume_smoothing_frames = config.VOLUME_SMOOTHING_FRAMES

        # Configure close-app hold duration from config
        self.action_executor.close_app_hold_duration = config.CLOSE_APP_HOLD_DURATION

        # Configure minimize-app hold duration from config
        self.action_executor.minimize_app_hold_duration = config.MINIMIZE_APP_HOLD_DURATION

        self.fps_counter = FPSCounter()

        # Get screen size for cursor control
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"OK Screen size detected: {self.screen_width}x{self.screen_height}")

        # Camera setup
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

        # State
        self.current_gesture = "No hand detected"
        self.current_confidence = 0.0

        # Black screen mode state
        self.black_screen_enabled = False
        self._load_black_screen_setting()

        # Window name
        self.window_name = "Compact Mode - Gesture Control"

    def _on_mouse_event(self, event, x, y, flags, param):
        """
        Handle mouse events for settings button interaction.
        Args:
            event: OpenCV mouse event type
            x, y: Mouse coords
            flags: Additional flags
            param: Additional params
        """
        btn_x, btn_y, btn_w, btn_h = self.settings_button_rect

        inside_button = (btn_x <= x <=btn_x + btn_w and
                         btn_y <= y <= btn_y + btn_h)

        self.settings_button_hovered = inside_button

        if event == cv2.EVENT_LBUTTONDOWN and inside_button:
            print("Settings button clicked")
            self._open_display_settings()

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

        cv2.setMouseCallback(self.window_name, self._on_mouse_event)
        # Set always-on-top
        try:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            print("OK Always-on-top enabled")
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

            # Detect hand (on original frame before blackening)
            hand_landmarks = self.detector.detect_hand(frame)

            # Replace with black frame if enabled (AFTER detection, BEFORE rendering)
            if self.black_screen_enabled:
                frame = np.zeros_like(frame)

            # Predict gesture
            if hand_landmarks:
                gesture, confidence, valid = self.recognizer.predict_smooth(
                    hand_landmarks, normalize_landmarks
                )

                if valid:
                    self.current_gesture = gesture
                    self.current_confidence = confidence

                    # Update action trigger (NOW PASSES hand_landmarks)
                    triggered_gesture = self.action_executor.update(gesture, hand_landmarks)

                    # Execute discrete action if triggered
                    if triggered_gesture:
                        self.action_executor.execute_action(triggered_gesture)

                    # Update continuous control (called every frame)
                    self.action_executor.update_continuous_control(
                        hand_landmarks=hand_landmarks,
                        sensitivity=self.config.CURSOR_SENSITIVITY,
                        smoothing=self.config.CURSOR_SMOOTHING,
                        screen_width=self.screen_width,
                        screen_height=self.screen_height
                    )

                    # Update drag control (called every frame)
                    self.action_executor.update_drag_control(
                        hand_landmarks=hand_landmarks,
                        sensitivity=self.config.CURSOR_SENSITIVITY,
                        smoothing=self.config.CURSOR_SMOOTHING,
                        screen_width=self.screen_width,
                        screen_height=self.screen_height
                    )

                    # Update volume control (called every frame)
                    self.action_executor.update_volume_control()

                    # Update scroll control (called every frame)
                    self.action_executor.update_scroll_control()

                    # Update close-app hold timer (called every frame)
                    self.action_executor.update_close_app()

                    # Update minimize-app hold timer (called every frame)
                    self.action_executor.update_minimize_app()

                    # Update swipe detection (called every frame when open_palm held)
                    swipe_triggered = self.action_executor.update_swipe_detection(
                        hand_landmarks=hand_landmarks,
                        screen_width=self.screen_width
                    )

                    # Toggle black screen if swipe detected
                    if swipe_triggered:
                        self._toggle_black_screen()
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

            # Handle keyboard input (with FPS limiting)
            wait_time = max(1, int(1000 / self.config.PROCESSING_FPS_LIMIT))
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                break

            if self.action_executor.should_close:
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

        # Show close-app countdown progress bar
        if self.action_executor.close_app_active and self.action_executor.close_app_start_time:
            elapsed = time.time() - self.action_executor.close_app_start_time
            progress = min(1.0, elapsed / self.action_executor.close_app_hold_duration)
            bar_width = int(width * progress)
            cv2.rectangle(frame, (0, height - 8), (bar_width, height), (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"CLOSING: {elapsed:.1f}s",
                (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )

        # Show minimize-app countdown progress bar
        if self.action_executor.minimize_app_active and self.action_executor.minimize_app_start_time:
            elapsed = time.time() - self.action_executor.minimize_app_start_time
            progress = min(1.0, elapsed / self.action_executor.minimize_app_hold_duration)
            bar_width = int(width * progress)
            cv2.rectangle(frame, (0, height - 8), (bar_width, height), (0, 165, 255), -1)
            cv2.putText(
                frame,
                f"MINIMIZING: {elapsed:.1f}s",
                (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 165, 255),
                1
            )

        # Show drag state indicator
        if self.action_executor.drag_active:
            cv2.putText(
                frame, "DRAGGING...", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2
            )

        # Show thumb-ring distance when cursor control active
        if (self.action_executor.continuous_active and
            self.action_executor.last_thumb_ring_distance is not None):
            distance = self.action_executor.last_thumb_ring_distance
            threshold = self.action_executor.proximity_threshold
            status = "READY" if distance < threshold else "EXTENDED"
            color = (0, 255, 0) if distance < threshold else (0, 100, 255)  # Green if ready, orange if extended

            cv2.putText(
                frame,
                f"Thumb: {distance:.3f} [{status}]",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # Show pinky-ring distance when cursor control active
        if (self.action_executor.continuous_active and
            self.action_executor.last_pinky_ring_distance is not None):
            distance = self.action_executor.last_pinky_ring_distance
            threshold = self.action_executor.proximity_double_click_threshold
            status = "READY" if distance < threshold else "EXTENDED"
            color = (255, 0, 255) if distance < threshold else (147, 20, 255)  # Magenta if ready, purple if extended

            cv2.putText(
                frame,
                f"Pinky: {distance:.3f} [{status}]",
                (10, 115),  # Position below thumb-ring display (y=95)
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # Show current volume when volume control active
        if self.action_executor.volume_control_active:
            current_volume = self.action_executor.get_current_volume_percent()
            if current_volume is not None:
                # Determine color based on volume level
                if current_volume > 75:
                    vol_color = (0, 100, 255)  # Orange (high)
                elif current_volume > 25:
                    vol_color = (0, 255, 0)    # Green (medium)
                else:
                    vol_color = (0, 255, 255)  # Yellow (low)

                gesture_text = "↑ VOL" if self.action_executor.volume_control_gesture == "volume_up" else "↓ VOL"

                cv2.putText(
                    frame,
                    f"{gesture_text}: {current_volume:.0f}%",
                    (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    vol_color,
                    2
                )

        # Display general volume level (always visible, not just during control)
        current_volume = self.action_executor.get_current_volume_percent()
        if current_volume is not None:
            cv2.putText(
                frame,
                f"Vol: {current_volume:.0f}%",
                (width - 90, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.config.COLOR_WHITE,
                1
            )

        btn_size = 30
        btn_x = width - 40
        btn_y = 10

        self.settings_button_rect = (btn_x, btn_y, btn_size, btn_size)

        center_x = btn_x + btn_size // 2
        center_y = btn_y + btn_size // 2
        radius = btn_size // 2

        if self.settings_button_hovered:
            btn_color = (255, 255, 255)
        else:
            btn_color = (200, 200, 200)

        cv2.circle(frame, (center_x, center_y), radius, btn_color, -1)

        text_x = btn_x + 7
        text_y = btn_y + 22

        cv2.putText(
            frame,
            "S",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (60, 60, 60),
            2
        )

    def _open_display_settings(self):
        """
        Open windows display settings page.
        """
        try:
            subprocess.Popen(['start', 'ms-settings:display'], shell=True)
            print("Windows display settings opened")
        except Exception as e:
            print(f"Failed: {e}")

    def _load_black_screen_setting(self):
        """
        Load black screen setting from persistent storage.

        If file doesn't exist or is corrupted, defaults to False (black screen OFF).
        """
        import json
        from pathlib import Path

        settings_path = self.config.PROJECT_ROOT / "data" / "app_settings.json"

        try:
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    self.black_screen_enabled = settings.get('black_screen_enabled', False)
                    print(f"OK Black screen setting loaded: {self.black_screen_enabled}")
            else:
                # File doesn't exist - use default
                self.black_screen_enabled = False
                print("OK Black screen setting file not found - using default (OFF)")
                # Create file with default settings
                self._save_black_screen_setting()

        except (json.JSONDecodeError, IOError) as e:
            # Corrupted file - use default and log warning
            print(f"WARNING: Failed to load black screen setting: {e}")
            print("  Using default (OFF)")
            self.black_screen_enabled = False
            # Overwrite corrupted file with valid default
            self._save_black_screen_setting()

    def _save_black_screen_setting(self):
        """
        Save black screen setting to persistent storage.

        Creates file if it doesn't exist. Handles write errors gracefully.
        """
        import json
        from pathlib import Path

        settings_path = self.config.PROJECT_ROOT / "data" / "app_settings.json"

        try:
            # Ensure data directory exists
            settings_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare settings data
            settings = {
                'black_screen_enabled': self.black_screen_enabled,
                'version': '1.0'
            }

            # Write to file with pretty formatting
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)

            print(f"OK Black screen setting saved: {self.black_screen_enabled}")

        except IOError as e:
            print(f"WARNING: Failed to save black screen setting: {e}")
            print("  Setting will not persist across sessions")

    def _toggle_black_screen(self):
        """
        Toggle black screen mode and provide visual feedback.

        Updates state, saves to persistent storage, and displays temporary message.
        """
        # Toggle state
        self.black_screen_enabled = not self.black_screen_enabled

        # Save to persistent storage
        self._save_black_screen_setting()

        # Trigger visual feedback via action executor
        status = "ON" if self.black_screen_enabled else "OFF"
        self.action_executor.last_action = f"Black Screen: {status}"
        self.action_executor.action_display_frames = 45  # ~1.5 seconds at 30fps

        print(f"OK Black screen toggled: {status}")

    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.detector.cleanup()

        print("\n" + "="*60)
        print("Compact Mode ended. Returning to menu...")
        print("="*60)
