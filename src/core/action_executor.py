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

    def reset(self):
        """Clear trigger state."""
        self.current_gesture = None
        self.frame_count = 0
        self.triggered_this_gesture = False

    def get_last_action(self):
        """Return the last executed action (for UI feedback)."""
        return self.last_action

    def decrement_display_frames(self):
        """Decrement action display counter (call each frame in UI rendering)."""
        if self.action_display_frames > 0:
            self.action_display_frames -= 1
