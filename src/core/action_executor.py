"""
Action Executor - Dwell-based trigger management and action execution.
"""

import time
import os
from datetime import datetime
from pathlib import Path
from pycaw.pycaw import AudioUtilities

import pyautogui

try:
    from pynput.mouse import Controller as PynputMouseController, Button as PynputButton
    from pynput.keyboard import Controller as KeyboardController, Key
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("WARNING: pynput not available - drag control disabled")


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

        # Proximity-based click state (for click-during-point)
        self.last_thumb_ring_distance = None      # Previous frame's distance
        self.proximity_click_triggered = False     # Debounce flag
        self.proximity_threshold = 0.1            # Normalized distance threshold (tune this)

        # Proximity-based double-click state (for double-click-during-point)
        self.last_pinky_ring_distance = None       # Previous frame's distance
        self.proximity_double_click_triggered = False  # Debounce flag
        self.proximity_double_click_threshold = 0.08   # Normalized distance threshold

        # Volume control state (for continuous timed increments)
        self.volume_interface = None
        self.volume_control_active = False           # Whether volume control is active
        self.volume_control_gesture = None           # Current volume gesture ('volume_up' or 'volume_down')
        self.volume_last_increment_time = None       # Timestamp of last volume increment
        self.volume_increment_interval = 0.5         # Seconds between increments (configurable)
        self.volume_increment_percent = 5.0          # Volume change per increment (configurable)
        self.volume_smoothing_counter = 0            # Frames since initial trigger
        self.volume_smoothing_frames = 3             # Frames to wait before continuous increments start

        # Screenshot settings
        self.screenshot_save_dir = None              # Will be set from config

        # Drag control state
        self.drag_active = False
        self.drag_gesture = None
        self.drag_button_pressed = False
        self.origin_drag_cursor_x = None
        self.origin_drag_cursor_y = None
        self.origin_drag_hand_x = None
        self.origin_drag_hand_y = None
        self.last_drag_cursor_x = None
        self.last_drag_cursor_y = None

        # Close-app hold state
        self.close_app_active = False          # Whether the hold timer is running
        self.close_app_start_time = None       # Timestamp when hold started
        self.close_app_hold_duration = 5.0     # Seconds to hold before closing (set from config)
        self.should_close = False              # Flag read by the window loop to break

        # Minimize-app hold state
        self.minimize_app_active = False       # Whether the hold timer is running
        self.minimize_app_start_time = None    # Timestamp when hold started
        self.minimize_app_hold_duration = 2.5  # Seconds to hold before minimizing (set from config)

        # Scroll control state (for continuous timed increments)
        self.scroll_control_active = False           # Whether scroll control is active
        self.scroll_control_gesture = None           # Current scroll gesture ('scroll_up' or 'scroll_down')
        self.scroll_last_increment_time = None       # Timestamp of last scroll increment
        self.scroll_increment_interval = 0.1         # Seconds between increments (faster than volume)
        self.scroll_smoothing_counter = 0            # Frames since initial trigger
        self.scroll_smoothing_frames = 3             # Frames to wait before continuous increments start

        # Swipe detection state (for black screen toggle)
        self.swipe_active = False                    # Whether swipe detection is active
        self.swipe_start_x = None                    # Starting X position of landmark 9 (normalized 0-1)
        self.swipe_start_y = None                    # Starting Y position for vertical tolerance check
        self.swipe_triggered = False                 # Debounce flag to prevent multiple triggers
        self.swipe_threshold = 0.5                   # Horizontal movement threshold (50% of screen width)
        self.swipe_vertical_tolerance = 0.1          # Max vertical movement to still count as horizontal swipe

        # Initialize pynput controllers
        if PYNPUT_AVAILABLE:
            self.mouse_controller = PynputMouseController()
            self.keyboard = KeyboardController()
        else:
            self.mouse_controller = None
            self.keyboard = None

        # Initialize Windows audio interface
        self._initialize_volume_control()

    def _initialize_volume_control(self):
        """
        Initialize Windows Core Audio API for volume control.

        Establishes connection to the default audio output device.
        Handles initialization failures gracefully.
        """
        try:
            # Get default audio output device (returns AudioDevice object)
            devices = AudioUtilities.GetSpeakers()
            # Get the IAudioEndpointVolume interface directly
            self.volume_interface = devices.EndpointVolume

            # Test access by getting current volume
            current_volume = self.volume_interface.GetMasterVolumeLevelScalar()
            print(f"OK Volume control initialized (current: {current_volume*100:.0f}%)")

        except Exception as e:
            print(f"WARNING: Volume control initialization failed: {e}")
            print("  Volume gestures will be disabled.")
            self.volume_interface = None

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
            if self.close_app_active:
                self._deactivate_close_app()
            if self.minimize_app_active:
                self._deactivate_minimize_app()
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
                elif detected_gesture == 'index_middle':
                    # Activate drag control
                    self._activate_drag_control(detected_gesture, hand_landmarks)
                    return None
                elif detected_gesture in self.gesture_actions and \
                     self.gesture_actions[detected_gesture] in ['volume_up', 'volume_down']:
                    # Activate continuous volume control
                    self._activate_volume_control(detected_gesture)
                    return None
                elif detected_gesture in self.gesture_actions and \
                     self.gesture_actions[detected_gesture] in ['scroll_up', 'scroll_down']:
                    # Activate continuous scroll control
                    self._activate_scroll_control(detected_gesture)
                    return None
                elif detected_gesture in self.gesture_actions and \
                     self.gesture_actions[detected_gesture] == 'close_app':
                    # Activate BOTH swipe detection and close-app hold timer
                    # The first to complete wins and deactivates the other
                    self._activate_swipe_detection(hand_landmarks)
                    self._activate_close_app()
                    return None
                elif detected_gesture in self.gesture_actions and \
                     self.gesture_actions[detected_gesture] == 'minimize_app':
                    # Activate minimize-app hold timer
                    self._activate_minimize_app()
                    return None
                else:
                    # Trigger discrete action as before
                    return detected_gesture  # TRIGGER ACTION
        else:
            # New gesture detected
            # Deactivate continuous control if active
            if self.continuous_active:
                self._deactivate_continuous_control()

            # Deactivate volume control if active
            if self.volume_control_active:
                self._deactivate_volume_control()

            # Deactivate scroll control if active
            if self.scroll_control_active:
                self._deactivate_scroll_control()

            # Deactivate drag control if active
            if self.drag_active:
                self._deactivate_drag_control()

            # Deactivate close-app hold if active
            if self.close_app_active:
                self._deactivate_close_app()

            # Deactivate minimize-app hold if active
            if self.minimize_app_active:
                self._deactivate_minimize_app()

            # Deactivate swipe detection if active
            if self.swipe_active:
                self._deactivate_swipe_detection()

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
        elif action_name == "volume_up":
            self._execute_volume_up()
        elif action_name == "volume_down":
            self._execute_volume_down()
        elif action_name == "close_app":
            self._execute_close_app()
        elif action_name == "toggle_pause_play":
            self._execute_toggle_pause_play()
        elif action_name == "previous_track":
            self._execute_previous_track()
        elif action_name == "next_track":
            self._execute_next_track()
        elif action_name == "screenshot":
            self._execute_screenshot()
        # Add more action handlers here as needed

    def _execute_left_click(self):
        """Execute left mouse click at current cursor position."""
        pyautogui.click()
        self.last_action = "LEFT CLICK"
        self.action_display_frames = 30  # Show feedback for ~1 second
        print(f"OK LEFT CLICK executed (gesture triggered)")

    def _execute_double_click(self):
        """Execute double mouse click at current cursor position."""
        pyautogui.doubleClick()
        self.last_action = "DOUBLE CLICK"
        self.action_display_frames = 30  # Show feedback for ~1 second
        print(f"OK DOUBLE CLICK executed (proximity triggered)")

    def _execute_right_click(self):
        """Execute right mouse click at current cursor position."""
        pyautogui.click(button='right')
        self.last_action = "RIGHT CLICK"
        self.action_display_frames = 30
        print(f"OK RIGHT CLICK executed (gesture triggered)")

    def _execute_screenshot(self):
        """
        Capture screenshot and save to configured directory.

        Saves with timestamp-based filename: screenshot_YYYYMMDD_HHMMSS.png
        Creates directory if it doesn't exist.
        """
        # Ensure save directory exists
        save_dir = Path(self.screenshot_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = save_dir / filename

        # Capture and save screenshot
        screenshot = pyautogui.screenshot()
        screenshot.save(str(filepath))

        # UI feedback
        self.last_action = f"SCREENSHOT: {filename}"
        self.action_display_frames = 60  # Show for ~2 seconds
        print(f"OK Screenshot saved: {filepath}")

    def _execute_scroll_up(self):
        """Scroll up by 100 units."""
        pyautogui.scroll(100)
        self.last_action = "SCROLL UP"
        self.action_display_frames = 30
        print(f"OK SCROLL UP executed (gesture triggered)")

    def _execute_scroll_down(self):
        """Scroll down by 100 units."""
        pyautogui.scroll(-100)
        self.last_action = "SCROLL DOWN"
        self.action_display_frames = 30
        print(f"OK SCROLL DOWN executed (gesture triggered)")

    def _execute_volume_up(self):
        """
        Increase system volume by configured increment.

        Clamps to maximum of 100% (1.0 scalar).
        """
        if self.volume_interface is None:
            print("WARNING: Volume control not available")
            return

        try:
            # Get current volume (0.0 - 1.0)
            current_volume = self.volume_interface.GetMasterVolumeLevelScalar()

            # Calculate new volume
            increment = self.volume_increment_percent / 100.0
            new_volume = min(1.0, current_volume + increment)

            # Set new volume
            self.volume_interface.SetMasterVolumeLevelScalar(new_volume, None)

            # Update UI feedback
            self.last_action = f"VOL UP ({new_volume*100:.0f}%)"
            self.action_display_frames = 30

            print(f"OK VOLUME UP: {current_volume*100:.0f}% -> {new_volume*100:.0f}%")

        except Exception as e:
            print(f"WARNING: Volume control error: {e}")

    def _execute_volume_down(self):
        """
        Decrease system volume by configured increment.

        Clamps to minimum of 0% (0.0 scalar).
        """
        if self.volume_interface is None:
            print("WARNING: Volume control not available")
            return

        try:
            # Get current volume (0.0 - 1.0)
            current_volume = self.volume_interface.GetMasterVolumeLevelScalar()

            # Calculate new volume
            decrement = self.volume_increment_percent / 100.0
            new_volume = max(0.0, current_volume - decrement)

            # Set new volume
            self.volume_interface.SetMasterVolumeLevelScalar(new_volume, None)

            # Update UI feedback
            self.last_action = f"VOL DOWN ({new_volume*100:.0f}%)"
            self.action_display_frames = 30

            print(f"OK VOLUME DOWN: {current_volume*100:.0f}% -> {new_volume*100:.0f}%")

        except Exception as e:
            print(f"WARNING: Volume control error: {e}")

    def _execute_toggle_pause_play(self):
        """
        Toggle global pause/play using Windows media key.

        Sends VK_MEDIA_PLAY_PAUSE key code which works globally across all media applications
        (Spotify, YouTube, VLC, Windows Media Player, etc.) regardless of focus.
        """
        if self.keyboard is None:
            print("WARNING: Keyboard control not available - pause/play disabled")
            return

        try:
            # Send global media play/pause key
            self.keyboard.press(Key.media_play_pause)
            self.keyboard.release(Key.media_play_pause)

            # Update UI feedback
            self.last_action = "PAUSE/PLAY"
            self.action_display_frames = 30

            print("OK PAUSE/PLAY toggled (global media key)")

        except Exception as e:
            print(f"WARNING: Media key control error: {e}")

    def _execute_previous_track(self):
        """
        Skip to previous track using Windows media key.

        Sends VK_MEDIA_PREV_TRACK key code which works globally across all media applications.
        """
        if self.keyboard is None:
            print("WARNING: Keyboard control not available - previous track disabled")
            return

        try:
            # Send global media previous key
            self.keyboard.press(Key.media_previous)
            self.keyboard.release(Key.media_previous)

            # Update UI feedback
            self.last_action = "PREVIOUS TRACK"
            self.action_display_frames = 30

            print("OK PREVIOUS TRACK (global media key)")

        except Exception as e:
            print(f"WARNING: Media key control error: {e}")

    def _execute_next_track(self):
        """
        Skip to next track using Windows media key.

        Sends VK_MEDIA_NEXT_TRACK key code which works globally across all media applications.
        """
        if self.keyboard is None:
            print("WARNING: Keyboard control not available - next track disabled")
            return

        try:
            # Send global media next key
            self.keyboard.press(Key.media_next)
            self.keyboard.release(Key.media_next)

            # Update UI feedback
            self.last_action = "NEXT TRACK"
            self.action_display_frames = 30

            print("OK NEXT TRACK (global media key)")

        except Exception as e:
            print(f"WARNING: Media key control error: {e}")

    def get_current_volume_percent(self):
        """
        Get current system volume as percentage (0-100).

        Returns:
            float: Volume percentage, or None if volume control unavailable
        """
        if self.volume_interface is None:
            return None

        try:
            volume_scalar = self.volume_interface.GetMasterVolumeLevelScalar()
            return volume_scalar * 100.0
        except Exception as e:
            print(f"WARNING: Error getting volume: {e}")
            return None

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

        # Reset proximity click state
        self.last_thumb_ring_distance = None
        self.proximity_click_triggered = False

        # Reset proximity double-click state
        self.last_pinky_ring_distance = None
        self.proximity_double_click_triggered = False

        print(f"OK Cursor control ACTIVATED")
        print(f"  Origin cursor: ({current_x}, {current_y})")
        print(f"  Origin hand: ({self.origin_hand_x:.3f}, {self.origin_hand_y:.3f})")

    def _activate_drag_control(self, gesture, hand_landmarks):
        """Activate drag control - press mouse button and capture origin positions."""
        if not PYNPUT_AVAILABLE or self.mouse_controller is None:
            print("ERROR: Cannot activate drag - pynput not available")
            return

        self.drag_active = True
        self.drag_gesture = gesture

        # Press and hold left mouse button
        try:
            self.mouse_controller.press(PynputButton.left)
            self.drag_button_pressed = True
            print(f"OK Drag control ACTIVATED (gesture: {gesture})")
            print(f"   Mouse button PRESSED")
        except Exception as e:
            print(f"ERROR: Failed to press mouse button: {e}")
            self.drag_active = False
            return

        # Capture origin positions
        current_x, current_y = pyautogui.position()
        self.origin_drag_cursor_x = current_x
        self.origin_drag_cursor_y = current_y
        self.last_drag_cursor_x = current_x
        self.last_drag_cursor_y = current_y

        if hand_landmarks:
            self.origin_drag_hand_x = hand_landmarks.landmark[8].x  # Index finger tip
            self.origin_drag_hand_y = hand_landmarks.landmark[8].y

        self.last_action = "DRAG START"
        self.action_display_frames = 30

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

        # Check for proximity-based click trigger
        current_distance = self._calculate_thumb_ring_distance(hand_landmarks)

        # Detect threshold crossing (AWAY from ring finger = click)
        if self.last_thumb_ring_distance is not None:
            # Click on proximity EXIT: distance was below threshold, now above
            if (self.last_thumb_ring_distance < self.proximity_threshold and
                current_distance >= self.proximity_threshold and
                not self.proximity_click_triggered):

                # Trigger click
                self._execute_left_click()
                self.proximity_click_triggered = True
                print(f"OK PROXIMITY CLICK (distance: {self.last_thumb_ring_distance:.3f} -> {current_distance:.3f})")

            # Reset debounce when thumb returns close to ring finger
            elif current_distance < self.proximity_threshold:
                self.proximity_click_triggered = False

        # Update distance for next frame
        self.last_thumb_ring_distance = current_distance

        # Check for proximity-based double-click trigger (pinky-ring)
        current_double_click_distance = self._calculate_pinky_ring_distance(hand_landmarks)

        # Detect threshold crossing (AWAY from ring finger = double-click)
        if self.last_pinky_ring_distance is not None:
            # Double-click on proximity EXIT: distance was below threshold, now above
            if (self.last_pinky_ring_distance < self.proximity_double_click_threshold and
                current_double_click_distance >= self.proximity_double_click_threshold and
                not self.proximity_double_click_triggered):

                # Trigger double-click
                self._execute_double_click()
                self.proximity_double_click_triggered = True
                print(f"OK PROXIMITY DOUBLE-CLICK (distance: {self.last_pinky_ring_distance:.3f} -> {current_double_click_distance:.3f})")

            # Reset debounce when pinky returns close to ring finger
            elif current_double_click_distance < self.proximity_double_click_threshold:
                self.proximity_double_click_triggered = False

        # Update distance for next frame
        self.last_pinky_ring_distance = current_double_click_distance

    def update_drag_control(self, hand_landmarks, sensitivity=1.5, smoothing=0.3,
                            screen_width=1920, screen_height=1080):
        """Update cursor position during drag (call every frame while dragging)."""
        if not self.drag_active or hand_landmarks is None:
            return

        # Get current hand position (index finger tip = landmark 8)
        current_hand_x = hand_landmarks.landmark[8].x
        current_hand_y = hand_landmarks.landmark[8].y

        # Calculate hand movement delta (in normalized coords 0-1)
        delta_x = current_hand_x - self.origin_drag_hand_x
        delta_y = current_hand_y - self.origin_drag_hand_y

        # Transform to screen pixels with sensitivity
        pixel_delta_x = delta_x * screen_width * sensitivity
        pixel_delta_y = delta_y * screen_height * sensitivity

        # Calculate new cursor position (relative to drag origin)
        new_cursor_x = self.origin_drag_cursor_x + pixel_delta_x
        new_cursor_y = self.origin_drag_cursor_y + pixel_delta_y

        # Apply exponential moving average for smoothing
        smoothed_x = self.last_drag_cursor_x * smoothing + new_cursor_x * (1 - smoothing)
        smoothed_y = self.last_drag_cursor_y * smoothing + new_cursor_y * (1 - smoothing)

        # Clamp to screen boundaries
        smoothed_x = max(0, min(smoothed_x, screen_width - 1))
        smoothed_y = max(0, min(smoothed_y, screen_height - 1))

        # Update cursor position (button still pressed)
        pyautogui.moveTo(int(smoothed_x), int(smoothed_y), duration=0)

        # Store for next frame
        self.last_drag_cursor_x = smoothed_x
        self.last_drag_cursor_y = smoothed_y

    def _deactivate_continuous_control(self):
        """Reset continuous control state."""
        if self.continuous_active:
            print(f"OK Cursor control DEACTIVATED")

        self.continuous_gesture = None
        self.continuous_active = False
        self.origin_cursor_x = None
        self.origin_cursor_y = None
        self.origin_hand_x = None
        self.origin_hand_y = None
        self.last_cursor_x = None
        self.last_cursor_y = None

        # Reset proximity click state
        self.last_thumb_ring_distance = None
        self.proximity_click_triggered = False

        # Reset proximity double-click state
        self.last_pinky_ring_distance = None
        self.proximity_double_click_triggered = False

    def _deactivate_drag_control(self):
        """Release mouse button and reset drag state."""
        if self.drag_active:
            # Release mouse button
            if self.drag_button_pressed and self.mouse_controller is not None:
                try:
                    self.mouse_controller.release(PynputButton.left)
                    print(f"   Mouse button RELEASED")
                except Exception as e:
                    print(f"ERROR: Failed to release mouse button: {e}")

            print(f"OK Drag control DEACTIVATED")
            self.last_action = "DRAG END"
            self.action_display_frames = 30

        # Reset all drag state
        self.drag_active = False
        self.drag_gesture = None
        self.drag_button_pressed = False
        self.origin_drag_cursor_x = None
        self.origin_drag_cursor_y = None
        self.origin_drag_hand_x = None
        self.origin_drag_hand_y = None
        self.last_drag_cursor_x = None
        self.last_drag_cursor_y = None

    def _activate_volume_control(self, gesture):
        """
        Activate continuous volume control for a volume gesture.

        Args:
            gesture: Name of the volume gesture ('thumbs_up' or 'thumbs_down')
        """
        if self.volume_interface is None:
            print("WARNING: Volume control not available")
            return

        action_name = self.gesture_actions.get(gesture)
        self.volume_control_active = True
        self.volume_control_gesture = action_name
        self.volume_last_increment_time = time.time()
        self.volume_smoothing_counter = 0

        # Execute first increment immediately
        if action_name == "volume_up":
            self._execute_volume_up()
        elif action_name == "volume_down":
            self._execute_volume_down()

        print(f"OK Volume control ACTIVATED: {action_name}")

    def _deactivate_volume_control(self):
        """Reset volume control state."""
        if self.volume_control_active:
            print(f"OK Volume control DEACTIVATED")

        self.volume_control_active = False
        self.volume_control_gesture = None
        self.volume_last_increment_time = None
        self.volume_smoothing_counter = 0

    def _activate_scroll_control(self, gesture):
        """
        Activate continuous scroll control for a scroll gesture.

        Args:
            gesture: Name of the scroll gesture ('thumbs_up' or 'thumbs_down')
        """
        action_name = self.gesture_actions.get(gesture)
        self.scroll_control_active = True
        self.scroll_control_gesture = action_name
        self.scroll_last_increment_time = time.time()
        self.scroll_smoothing_counter = 0

        # Execute first scroll immediately
        if action_name == "scroll_up":
            self._execute_scroll_up()
        elif action_name == "scroll_down":
            self._execute_scroll_down()

        print(f"OK Scroll control ACTIVATED: {action_name}")

    def _deactivate_scroll_control(self):
        """Reset scroll control state."""
        if self.scroll_control_active:
            print(f"OK Scroll control DEACTIVATED")

        self.scroll_control_active = False
        self.scroll_control_gesture = None
        self.scroll_last_increment_time = None
        self.scroll_smoothing_counter = 0

    def _activate_swipe_detection(self, hand_landmarks):
        """
        Capture starting position of landmark 9 (middle finger MCP) for swipe detection.

        Args:
            hand_landmarks: MediaPipe hand landmarks
        """
        self.swipe_active = True
        self.swipe_triggered = False

        if hand_landmarks:
            # Landmark 9 = middle finger MCP joint
            self.swipe_start_x = hand_landmarks.landmark[9].x  # Normalized (0-1)
            self.swipe_start_y = hand_landmarks.landmark[9].y
            print(f"OK Swipe detection ACTIVATED (start X: {self.swipe_start_x:.3f}, Y: {self.swipe_start_y:.3f})")

    def _activate_close_app(self):
        """Start the close-app hold timer."""
        self.close_app_active = True
        self.close_app_start_time = time.time()
        self.last_action = "CLOSING APP..."
        self.action_display_frames = 30
        print(f"OK Close-app hold ACTIVATED (hold for {self.close_app_hold_duration:.0f}s)")

    def _deactivate_close_app(self):
        """Cancel the close-app hold timer."""
        if self.close_app_active:
            print("OK Close-app hold DEACTIVATED (gesture released)")
        self.close_app_active = False
        self.close_app_start_time = None

    def _deactivate_swipe_detection(self):
        """Reset swipe detection state."""
        if self.swipe_active:
            print("OK Swipe detection DEACTIVATED")

        self.swipe_active = False
        self.swipe_start_x = None
        self.swipe_start_y = None
        self.swipe_triggered = False

    def update_close_app(self):
        """
        Check hold duration and set should_close flag when threshold is met.

        Call this every frame when close-app hold may be active.
        """
        if not self.close_app_active or self.close_app_start_time is None:
            return
        elapsed = time.time() - self.close_app_start_time
        if elapsed >= self.close_app_hold_duration:
            self._execute_close_app()

    def _execute_close_app(self):
        """Signal the application to close."""
        self._deactivate_swipe_detection()  # Close-app won the race
        self.should_close = True
        self.close_app_active = False
        self.last_action = "CLOSING APP"
        self.action_display_frames = 30
        print("OK CLOSE APP executed (open_palm held 5s)")

    def _activate_minimize_app(self):
        """Start the minimize-app hold timer."""
        self.minimize_app_active = True
        self.minimize_app_start_time = time.time()
        self.last_action = "MINIMIZING..."
        self.action_display_frames = 30
        print(f"OK Minimize-app hold ACTIVATED (hold for {self.minimize_app_hold_duration:.1f}s)")

    def _deactivate_minimize_app(self):
        """Cancel the minimize-app hold timer."""
        if self.minimize_app_active:
            print("OK Minimize-app hold DEACTIVATED (gesture released)")
        self.minimize_app_active = False
        self.minimize_app_start_time = None

    def update_minimize_app(self):
        """
        Check hold duration and minimize the focused app when threshold is met.

        Call this every frame when minimize-app hold may be active.
        """
        if not self.minimize_app_active or self.minimize_app_start_time is None:
            return
        elapsed = time.time() - self.minimize_app_start_time
        if elapsed >= self.minimize_app_hold_duration:
            self._execute_minimize_app()

    def _execute_minimize_app(self):
        """Minimize the currently focused application window."""
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.user32.ShowWindow(hwnd, 6)  # 6 = SW_MINIMIZE
        self.minimize_app_active = False
        self.minimize_app_start_time = None
        self.last_action = "MINIMIZED"
        self.action_display_frames = 30
        print("OK MINIMIZE APP executed (fist held 2.5s)")

    def update_volume_control(self):
        """
        Update volume control - execute timed increments.

        Call this every frame when volume control is active.
        Checks elapsed time and triggers increments at configured interval.
        """
        if not self.volume_control_active or self.volume_interface is None:
            return

        # Increment smoothing counter
        self.volume_smoothing_counter += 1

        # Wait for smoothing period before starting continuous increments
        if self.volume_smoothing_counter < self.volume_smoothing_frames:
            return

        # Check if enough time has elapsed since last increment
        current_time = time.time()
        elapsed = current_time - self.volume_last_increment_time

        if elapsed >= self.volume_increment_interval:
            # Execute volume change
            if self.volume_control_gesture == "volume_up":
                self._execute_volume_up()
            elif self.volume_control_gesture == "volume_down":
                self._execute_volume_down()

            # Update last increment timestamp
            self.volume_last_increment_time = current_time

    def update_scroll_control(self):
        """
        Update scroll control - execute timed increments.

        Call this every frame when scroll control is active.
        Checks elapsed time and triggers increments at configured interval.
        """
        if not self.scroll_control_active:
            return

        # Increment smoothing counter
        self.scroll_smoothing_counter += 1

        # Wait for smoothing period before starting continuous increments
        if self.scroll_smoothing_counter < self.scroll_smoothing_frames:
            return

        # Check if enough time has elapsed since last increment
        current_time = time.time()
        elapsed = current_time - self.scroll_last_increment_time

        if elapsed >= self.scroll_increment_interval:
            # Execute scroll
            if self.scroll_control_gesture == "scroll_up":
                self._execute_scroll_up()
            elif self.scroll_control_gesture == "scroll_down":
                self._execute_scroll_down()

            # Update last increment timestamp
            self.scroll_last_increment_time = current_time

    def update_swipe_detection(self, hand_landmarks, screen_width):
        """
        Check for horizontal swipe movement when open_palm is held.

        Args:
            hand_landmarks: Current hand landmarks
            screen_width: Screen width in pixels (used for threshold calculation)

        Returns:
            bool: True if swipe threshold was met (triggers black screen toggle)
        """
        if not self.swipe_active or self.swipe_triggered or hand_landmarks is None:
            return False

        # Get current position of landmark 9 (middle finger MCP)
        current_x = hand_landmarks.landmark[9].x
        current_y = hand_landmarks.landmark[9].y

        # Calculate horizontal and vertical movement (in normalized coords 0-1)
        delta_x = abs(current_x - self.swipe_start_x)
        delta_y = abs(current_y - self.swipe_start_y)

        # Check if movement is primarily horizontal
        if delta_y > self.swipe_vertical_tolerance:
            # Too much vertical movement - not a horizontal swipe
            # Reset start position to allow course correction
            self.swipe_start_x = current_x
            self.swipe_start_y = current_y
            return False

        # Check if horizontal movement exceeds threshold (50% of screen width)
        if delta_x >= self.swipe_threshold:
            # Swipe detected! Trigger black screen toggle
            self.swipe_triggered = True
            direction = "RIGHT" if current_x > self.swipe_start_x else "LEFT"
            print(f"OK SWIPE {direction} DETECTED (delta X: {delta_x:.3f})")

            # Deactivate close-app hold since swipe won the race
            self._deactivate_close_app()
            self._deactivate_swipe_detection()

            return True  # Signal to toggle black screen

        return False

    def _calculate_thumb_ring_distance(self, hand_landmarks):
        """
        Calculate normalized Euclidean distance between thumb tip and ring finger DIP joint.

        Args:
            hand_landmarks: MediaPipe hand landmarks

        Returns:
            float: Normalized distance (0-1 scale)
        """
        import math

        # Landmark 4 = thumb tip
        # Landmark 15 = ring finger DIP joint (second-to-tip joint)
        thumb_tip = hand_landmarks.landmark[4]
        ring_dip = hand_landmarks.landmark[15]

        # Euclidean distance in normalized coordinates
        distance = math.sqrt(
            (thumb_tip.x - ring_dip.x) ** 2 +
            (thumb_tip.y - ring_dip.y) ** 2
        )

        return distance

    def _calculate_pinky_ring_distance(self, hand_landmarks):
        """
        Calculate normalized Euclidean distance between pinky tip and ring finger tip.

        Args:
            hand_landmarks: MediaPipe hand landmarks

        Returns:
            float: Normalized distance (0-1 scale)
        """
        import math

        # Landmark 20 = pinky tip
        # Landmark 16 = ring finger tip
        pinky_tip = hand_landmarks.landmark[20]
        ring_tip = hand_landmarks.landmark[16]

        # Euclidean distance in normalized coordinates
        distance = math.sqrt(
            (pinky_tip.x - ring_tip.x) ** 2 +
            (pinky_tip.y - ring_tip.y) ** 2
        )

        return distance

    def reset(self):
        """Clear all state (discrete, continuous, volume, and drag control)."""
        # Discrete action state
        self.current_gesture = None
        self.frame_count = 0
        self.triggered_this_gesture = False

        # Continuous control state
        if self.continuous_active:
            self._deactivate_continuous_control()

        # Volume control state
        if self.volume_control_active:
            self._deactivate_volume_control()

        # Drag control state
        if self.drag_active:
            self._deactivate_drag_control()

        # Close-app hold state
        if self.close_app_active:
            self._deactivate_close_app()
        self.should_close = False

        # Minimize-app hold state
        if self.minimize_app_active:
            self._deactivate_minimize_app()

        # Swipe detection state
        if self.swipe_active:
            self._deactivate_swipe_detection()

    def get_last_action(self):
        """Return the last executed action (for UI feedback)."""
        return self.last_action

    def decrement_display_frames(self):
        """Decrement action display counter (call each frame in UI rendering)."""
        if self.action_display_frames > 0:
            self.action_display_frames -= 1
