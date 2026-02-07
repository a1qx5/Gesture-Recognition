"""
FPS Counter - Calculate frames per second for performance monitoring.
"""

import time


class FPSCounter:
    """
    Simple FPS (frames per second) counter for performance monitoring.

    Usage:
        fps_counter = FPSCounter()
        while True:
            # ... process frame ...
            fps_counter.update()
            fps = fps_counter.get_fps()
    """

    def __init__(self):
        """Initialize FPS counter."""
        self.last_time = time.time()
        self.fps = 0.0

    def update(self):
        """Update FPS calculation (call once per frame)."""
        current_time = time.time()
        delta_time = current_time - self.last_time

        # Prevent division by zero
        if delta_time > 0:
            self.fps = 1.0 / delta_time

        self.last_time = current_time

    def get_fps(self):
        """Get current FPS value."""
        return self.fps
