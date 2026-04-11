"""
UI Utilities - Shared drawing functions for OpenCV displays.
"""

import cv2
import numpy as np


def get_gesture_color(gesture, config):
    """
    Get color for gesture based on its status.

    Args:
        gesture: Gesture name or status string
        config: AppConfig instance with color definitions

    Returns:
        tuple: BGR color tuple
    """
    if gesture == "No hand detected":
        return config.COLOR_GRAY
    elif "Invalid" in gesture:
        return config.COLOR_RED
    elif gesture == "null":
        return config.COLOR_LIGHT_GRAY
    else:
        return config.COLOR_GREEN  # Active gesture


def draw_semi_transparent_panel(frame, height, alpha=0.6, color=(0, 0, 0)):
    """
    Draw semi-transparent panel at top of frame.

    Args:
        frame: Image to draw on (modified in-place)
        height: Height of panel in pixels
        alpha: Transparency (0=transparent, 1=opaque)
        color: Panel color in BGR
    """
    width = frame.shape[1]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_gesture_label(frame, gesture, confidence, position, config, font_scale=1.2):
    """
    Draw gesture name with confidence at specified position.

    Args:
        frame: Image to draw on
        gesture: Gesture name or status
        confidence: Confidence score (0-1)
        position: (x, y) tuple for text position
        config: AppConfig instance
        font_scale: Size of text
    """
    color = get_gesture_color(gesture, config)
    text = f"Current Gesture: {gesture}"

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        3
    )


def draw_confidence(frame, confidence, position, config, font_scale=0.7):
    """
    Draw confidence percentage.

    Args:
        frame: Image to draw on
        confidence: Confidence score (0-1)
        position: (x, y) tuple for text position
        config: AppConfig instance
        font_scale: Size of text
    """
    text = f"Confidence: {confidence*100:.1f}%"

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        config.COLOR_WHITE,
        2
    )


def draw_fps_counter(frame, fps, config):
    """
    Draw FPS counter in top-right corner.

    Args:
        frame: Image to draw on
        fps: Current FPS value
        config: AppConfig instance
    """
    height, width = frame.shape[:2]
    text = f"FPS: {fps:.1f}"

    # Calculate text size to position in top-right
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    position = (width - text_size[0] - 20, 30)

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        config.COLOR_YELLOW,
        2
    )


def draw_action_feedback(frame, action_executor, config):
    """
    Draw action feedback (e.g., "CLICK!") when action is triggered.

    Args:
        frame: Image to draw on
        action_executor: ActionExecutor instance
        config: AppConfig instance
    """
    if action_executor.action_display_frames > 0:
        action_text = f"ACTION: {action_executor.last_action}"

        # Pulsing effect based on remaining frames
        alpha = min(1.0, action_executor.action_display_frames / 15.0)
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


def draw_scale_diagnostics(frame, result, position, config):
    """
    Draw scale normalization diagnostics.

    Args:
        frame: Image to draw on
        result: Normalization result dict from normalize_landmarks()
        position: (x, y) tuple for text position
        config: AppConfig instance
    """
    if result is None:
        return

    scale_color = config.COLOR_GREEN if result['valid'] else config.COLOR_RED

    text = f"Scale: {result['scale_used']:.4f} (W-MCP: {result['scale_wrist_mcp']:.4f}, Palm: {result['scale_palm_width']:.4f})"

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        scale_color,
        2
    )


def draw_instructions(frame, instructions, config):
    """
    Draw instruction text at bottom of frame.

    Args:
        frame: Image to draw on
        instructions: Instruction string to display
        config: AppConfig instance
    """
    height = frame.shape[0]

    cv2.putText(
        frame,
        instructions,
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        config.COLOR_LIGHT_GRAY,
        1
    )
