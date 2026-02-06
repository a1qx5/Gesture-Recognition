class HybridTrigger:
    def __init__(self, min_dwell=5):
        self.min_dwell = min_dwell  # ~0.15s at 30 fps
        self.current_gesture = None
        self.frame_count = 0
        self.triggered_this_gesture = False
        
    def update(self, detected_gesture):
        """Trigger once per gesture hold after minimum dwell"""
        
        if detected_gesture == self.current_gesture:
            self.frame_count += 1
            
            # Trigger only once when dwell threshold first met
            if self.frame_count == self.min_dwell and not self.triggered_this_gesture:
                self.triggered_this_gesture = True
                return detected_gesture  # TRIGGER ACTION
                
        else:
            # New gesture - reset
            self.current_gesture = detected_gesture
            self.frame_count = 0
            self.triggered_this_gesture = False
            
        return None