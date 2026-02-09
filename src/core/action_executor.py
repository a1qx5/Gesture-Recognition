"""
Action Executor - Dwell-based trigger management and action execution.
"""

import pyautogui


class ActionExecutor:
    """
    Manages gesture-to-action triggering with dwell-based logic.

    Responsibilities:
    - Track gesture duration (dwell time)
    - Trigger actions only after minimum hold time
    - Prevent re-triggering while gesture is held
    - Execute system-level actions via PyAutoGUI
    """

    def __init__(self, gesture_actions, min_dwell_frames=5):
        """
        Initialize the action executor.

        Args:
            gesture_actions: Dict mapping gesture names to action names
            min_dwell_frames: Minimum frames to hold gesture before triggering (default: 5 ≈ 0.15s at 30fps)
        """
        self.gesture_actions = gesture_actions
        self.min_dwell_frames = min_dwell_frames

        # Trigger state
        self.current_gesture = None
        self.frame_count = 0
        self.triggered_this_gesture = False

        # UI feedback
        self.last_action = None
        self.action_display_frames = 0

        # Continuous control state (for cursor movement)
        self.continuous_gesture = None           # Currently active continuous gesture ('point')
        self.continuous_active = False           # Whether continuous control is active
        self.origin_cursor_x = None              # Screen cursor position at activation
        self.origin_cursor_y = None
        self.origin_hand_x = None                # Hand position (normalized 0-1) at activation
        self.origin_hand_y = None
        self.last_cursor_x = None                # For exponential smoothing
        self.last_cursor_y = None

    def update(self, detected_gesture, hand_landmarks=None):
        """
        Update trigger state with current gesture.

        Args:
            detected_gesture: The gesture detected this frame
            hand_landmarks: MediaPipe hand landmarks (needed for continuous control)

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

                # Check if this is a continuous control gesture
                if detected_gesture in ['point']:
                    # Activate continuous control (don't trigger discrete action)
                    self._activate_continuous_control(detected_gesture, hand_landmarks)
                    return None
                else:
                    # Trigger discrete action as before
                    return detected_gesture  # TRIGGER ACTION
        else:
            # New gesture detected
            # Deactivate continuous control if active
            if self.continuous_active:
                self._deactivate_continuous_control()

            # Reset state
            self.current_gesture = detected_gesture
            self.frame_count = 0
            self.triggered_this_gesture = False

        return None

    def execute_action(self, gesture):
        """
        Execute action based on triggered gesture.

        Args:
            gesture: Name of the gesture that triggered the action
        """
        # Look up action for this gesture
        action_name = self.gesture_actions.get(gesture)

        if action_name is None:
            return  # No action mapped for this gesture

        # Execute corresponding action
        if action_name == "left_click":
            self._execute_left_click()
        elif action_name == "right_click":
            self._execute_right_click()
        elif action_name == "scroll_up":
            self._execute_scroll_up()
        elif action_name == "scroll_down":
            self._execute_scroll_down()
        # Add more action handlers here as needed

    def _execute_left_click(self):
        """Execute left mouse click at current cursor position."""
        pyautogui.click()
        self.last_action = "LEFT CLICK"
        self.action_display_frames = 30  # Show feedback for ~1 second
        print(f"✓ LEFT CLICK executed (gesture triggered)")

    def _execute_right_click(self):
        """Execute right mouse click at current cursor position."""
        pyautogui.click(button='right')
        self.last_action = "RIGHT CLICK"
        self.action_display_frames = 30
        print(f"✓ RIGHT CLICK executed (gesture triggered)")

    def _execute_scroll_up(self):
        """Scroll up by 100 units."""
        pyautogui.scroll(100)
        self.last_action = "SCROLL UP"
        self.action_display_frames = 30
        print(f"✓ SCROLL UP executed (gesture triggered)")

    def _execute_scroll_down(self):
        """Scroll down by 100 units."""
        pyautogui.scroll(-100)
        self.last_action = "SCROLL DOWN"
        self.action_display_frames = 30
        print(f"✓ SCROLL DOWN executed (gesture triggered)")

    def _activate_continuous_control(self, gesture, hand_landmarks):
        """
        Capture origin positions when continuous control activates.

        Args:
            gesture: Name of continuous gesture ('point')
            hand_landmarks: MediaPipe hand landmarks
        """
        self.continuous_gesture = gesture
        self.continuous_active = True

        # Capture current screen cursor position
        current_x, current_y = pyautogui.position()
        self.origin_cursor_x = current_x
        self.origin_cursor_y = current_y
        self.last_cursor_x = current_x
        self.last_cursor_y = current_y

        # Capture current hand position (index finger tip = landmark 8)
        if hand_landmarks:
            self.origin_hand_x = hand_landmarks.landmark[8].x  # Normalized (0-1)
            self.origin_hand_y = hand_landmarks.landmark[8].y

        print(f"✓ Cursor control ACTIVATED")
        print(f"  Origin cursor: ({current_x}, {current_y})")
        print(f"  Origin hand: ({self.origin_hand_x:.3f}, {self.origin_hand_y:.3f})")

    def update_continuous_control(self, hand_landmarks, sensitivity=1.5, smoothing=0.3,
                                   screen_width=1920, screen_height=1080):
        """
        Update cursor position based on hand movement (call every frame).

        Args:
            hand_landmarks: Current hand landmarks
            sensitivity: Movement gain factor (pixels per normalized unit)
            smoothing: Exponential smoothing factor (0=none, 1=full)
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        if not self.continuous_active or hand_landmarks is None:
            return

        # Extract current hand position (landmark 8 = index finger tip)
        current_hand_x = hand_landmarks.landmark[8].x
        current_hand_y = hand_landmarks.landmark[8].y

        # Calculate hand movement delta (in normalized coords 0-1)
        delta_x = current_hand_x - self.origin_hand_x
        delta_y = current_hand_y - self.origin_hand_y

        # Transform to screen pixels with sensitivity
        pixel_delta_x = delta_x * screen_width * sensitivity
        pixel_delta_y = delta_y * screen_height * sensitivity

        # Calculate new cursor position relative to origin
        new_cursor_x = self.origin_cursor_x + pixel_delta_x
        new_cursor_y = self.origin_cursor_y + pixel_delta_y

        # Apply exponential moving average for smoothing
        smoothed_x = self.last_cursor_x * smoothing + new_cursor_x * (1 - smoothing)
        smoothed_y = self.last_cursor_y * smoothing + new_cursor_y * (1 - smoothing)

        # Clamp to screen boundaries
        smoothed_x = max(0, min(screen_width - 1, smoothed_x))
        smoothed_y = max(0, min(screen_height - 1, smoothed_y))

        # Move cursor (duration=0 for instant movement)
        pyautogui.moveTo(int(smoothed_x), int(smoothed_y), duration=0)

        # Update smoothing state
        self.last_cursor_x = smoothed_x
        self.last_cursor_y = smoothed_y

    def _deactivate_continuous_control(self):
        """Reset continuous control state."""
        if self.continuous_active:
            print(f"✓ Cursor control DEACTIVATED")

        self.continuous_gesture = None
        self.continuous_active = False
        self.origin_cursor_x = None
        self.origin_cursor_y = None
        self.origin_hand_x = None
        self.origin_hand_y = None
        self.last_cursor_x = None
        self.last_cursor_y = None

    def reset(self):
        """Clear all state (both discrete and continuous)."""
        # Discrete action state
        self.current_gesture = None
        self.frame_count = 0
        self.triggered_this_gesture = False

        # Continuous control state
        if self.continuous_active:
            self._deactivate_continuous_control()

    def get_last_action(self):
        """Return the last executed action (for UI feedback)."""
        return self.last_action

    def decrement_display_frames(self):
        """Decrement action display counter (call each frame in UI rendering)."""
        if self.action_display_frames > 0:
            self.action_display_frames -= 1
