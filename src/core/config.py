"""
Configuration - Centralized application settings.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    """
    Application configuration for hand gesture recognition.

    All paths, parameters, and settings are centralized here for easy modification.
    """

    # ========== Paths ==========
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)

    @property
    def MODEL_PATH(self):
        """Path to trained gesture classifier model."""
        return self.PROJECT_ROOT / "models" / "gesture_classifier_latest.pkl"

    @property
    def GESTURE_MAP_PATH(self):
        """Path to gesture ID -> name mapping JSON."""
        return self.PROJECT_ROOT / "data" / "gesture_map.json"

    # ========== Camera Settings ==========
    CAMERA_INDEX: int = 1
    CAMERA_WIDTH: int = 1280
    CAMERA_HEIGHT: int = 720
    CAMERA_FPS: int = 90

    # ========== MediaPipe Settings ==========
    MIN_DETECTION_CONFIDENCE: float = 0.7
    MIN_TRACKING_CONFIDENCE: float = 0.7

    # ========== Recognition Settings ==========
    SMOOTHING_HISTORY_SIZE: int = 5           # Frames to average for temporal smoothing
    NORMALIZATION_THRESHOLD: float = 0.05     # Minimum scale threshold for normalization

    # ========== Action Settings ==========
    ACTION_DWELL_FRAMES: int = 5              # Frames to hold gesture before triggering (~0.15s at 30fps)

    # ========== Cursor Control Settings ==========
    CURSOR_SENSITIVITY: float = 1.5           # Movement gain factor
    CURSOR_SMOOTHING: float = 0             # Exponential smoothing (0=none, 1=full)

    # ========== Volume Control Settings ==========
    VOLUME_INCREMENT_PERCENT: float = 5.0         # Volume change per step (0-100%)
    VOLUME_INCREMENT_INTERVAL: float = 0.5        # Seconds between increments
    VOLUME_SMOOTHING_FRAMES: int = 3             # Frames before continuous increments start

    # ========== Performance Settings ==========
    PROCESSING_FPS_LIMIT: int = 90            # Maximum processing FPS (should match CAMERA_FPS)

    # ========== Window Sizes ==========
    TESTING_MODE_SIZE: tuple = (1280, 720)    # Large window for testing
    COMPACT_MODE_SIZE: tuple = (320, 240)     # Small window for compact mode

    # ========== Gesture-to-Action Mappings ==========
    GESTURE_ACTIONS: dict = field(default_factory=lambda: {
        # "L_shape": "left_click",
        # Add more mappings here:
        # "fist": "right_click",
        "thumbs_up": "volume_up",
        "thumbs_down": "volume_down",
    })

    # ========== UI Colors (BGR format for OpenCV) ==========
    COLOR_GREEN: tuple = (0, 255, 0)          # Active gesture
    COLOR_GRAY: tuple = (128, 128, 128)       # No hand detected
    COLOR_LIGHT_GRAY: tuple = (200, 200, 200) # Null gesture
    COLOR_RED: tuple = (0, 0, 255)            # Invalid/error
    COLOR_YELLOW: tuple = (0, 255, 255)       # FPS counter, info
    COLOR_WHITE: tuple = (255, 255, 255)      # General text
    COLOR_BLACK: tuple = (0, 0, 0)            # Background overlay

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure project root exists
        if not self.PROJECT_ROOT.exists():
            raise ValueError(f"Project root does not exist: {self.PROJECT_ROOT}")

        # Ensure critical paths exist
        if not self.GESTURE_MAP_PATH.exists():
            raise FileNotFoundError(f"Gesture map not found: {self.GESTURE_MAP_PATH}")

        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {self.MODEL_PATH}")
