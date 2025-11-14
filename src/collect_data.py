"""
Data Collection UI for Hand Gesture Recognition
Captures webcam video, detects hands with MediaPipe, normalizes landmarks, and saves to CSV.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from pathlib import Path
import threading
import time

from normalize import normalize_landmarks, get_average_confidence


class GestureDataCollector:
    def __init__(self):
        # Load gesture map
        self.gesture_map = self.load_gesture_map()
        self.gesture_names = [self.gesture_map[str(i)] for i in sorted([int(k) for k in self.gesture_map.keys()])]
        
        # Current state
        self.current_gesture_id = 0
        self.current_gesture_name = self.gesture_map["0"]
        self.samples = []
        self.sample_counts = {name: 0 for name in self.gesture_names}
        
        # Load existing data counts from all CSV files
        self.existing_counts = self.load_existing_counts()
        
        # Auto-capture state
        self.auto_capture_active = False
        self.auto_capture_interval = 0.2  # Capture every 200ms during auto-capture
        self.last_capture_time = 0
        
        # Status for visual feedback
        self.last_capture_status = None  # "success", "failed", or None
        self.status_time = 0
        self.status_duration = 0.5  # Show status for 500ms
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Only detect one hand
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        
        # GUI setup
        self.setup_gui()
        
        # For thread-safe GUI updates
        self.running = True
        
    def load_gesture_map(self) -> dict:
        """Load gesture map from JSON file."""
        gesture_map_path = Path(__file__).parent.parent / "data" / "gesture_map.json"
        with open(gesture_map_path, 'r') as f:
            return json.load(f)
    
    def load_existing_counts(self) -> dict:
        """Load existing sample counts from the main CSV file."""
        counts = {name: 0 for name in self.gesture_names}
        data_dir = Path(__file__).parent.parent / "data"
        main_csv = data_dir / "gestures_data.csv"
        
        # Check if main CSV exists
        if not main_csv.exists():
            return counts
        
        # Load and count samples
        try:
            df = pd.read_csv(main_csv)
            # Count by gesture_id and map to gesture_name
            for gesture_id, count in df['gesture_id'].value_counts().items():
                gesture_name = self.gesture_map.get(str(gesture_id))
                if gesture_name and gesture_name in counts:
                    counts[gesture_name] += count
        except Exception as e:
            print(f"Warning: Could not load {main_csv.name}: {e}")
        
        return counts
    
    def setup_gui(self):
        """Set up tkinter GUI for gesture selection."""
        self.root = tk.Tk()
        self.root.title("Gesture Data Collector - Controls")
        self.root.geometry("450x400+20+50")  # Larger to fit counts display
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="Hand Gesture Data Collector", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Gesture selection
        gesture_frame = tk.Frame(self.root)
        gesture_frame.pack(pady=10)
        
        gesture_label = tk.Label(gesture_frame, text="Select Gesture:", font=("Arial", 12))
        gesture_label.pack(side=tk.LEFT, padx=5)
        
        self.gesture_var = tk.StringVar(value=self.current_gesture_name)
        self.gesture_dropdown = ttk.Combobox(
            gesture_frame,
            textvariable=self.gesture_var,
            values=self.gesture_names,
            state="readonly",
            width=20,
            font=("Arial", 11)
        )
        self.gesture_dropdown.pack(side=tk.LEFT, padx=5)
        self.gesture_dropdown.bind("<<ComboboxSelected>>", self.on_gesture_change)
        
        # Sample counts display
        self.counts_label = tk.Label(
            self.root,
            text=self.get_counts_text(),
            font=("Arial", 10),
            justify=tk.LEFT
        )
        self.counts_label.pack(pady=10)
        
        # Instructions
        instructions = """
Instructions:
• SPACE: Start/Stop auto-capture (batch mode)
• ENTER: Capture single sample
• Q or ESC: Quit and save data

Auto-capture will collect samples every 0.2s
while you hold or vary the gesture slightly.
        """
        instructions_label = tk.Label(
            self.root,
            text=instructions,
            font=("Arial", 9),
            justify=tk.LEFT,
            bg="#f0f0f0",
            relief=tk.RIDGE,
            padx=10,
            pady=10
        )
        instructions_label.pack(pady=10, padx=10, fill=tk.BOTH)
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Ready to collect data",
            font=("Arial", 10, "bold"),
            fg="blue"
        )
        self.status_label.pack(pady=5)
        
        # Quit button
        quit_button = tk.Button(
            self.root,
            text="Save & Quit",
            command=self.quit_application,
            font=("Arial", 11),
            bg="#ff6b6b",
            fg="white",
            padx=20,
            pady=5
        )
        quit_button.pack(pady=10)
        
    def on_gesture_change(self, event):
        """Handle gesture selection change."""
        selected_name = self.gesture_var.get()
        # Find the ID for this name
        for gesture_id, name in self.gesture_map.items():
            if name == selected_name:
                self.current_gesture_id = int(gesture_id)
                self.current_gesture_name = name
                break
        self.update_status(f"Switched to: {selected_name}")
        
    def get_counts_text(self) -> str:
        """Generate text showing sample counts for each gesture."""
        lines = ["Sample Counts (Existing + New):"]
        total_existing = sum(self.existing_counts.values())
        total_new = sum(self.sample_counts.values())
        
        for name in self.gesture_names:
            existing = self.existing_counts[name]
            new = self.sample_counts[name]
            total = existing + new
            lines.append(f"  {name}: {total} ({existing} + {new})")
        
        lines.append(f"\nTotal: {total_existing + total_new} ({total_existing} + {total_new})")
        return "\n".join(lines)
    
    def update_counts_display(self):
        """Update the counts label in GUI."""
        self.counts_label.config(text=self.get_counts_text())
    
    def update_status(self, message: str, color: str = "blue"):
        """Update status label in GUI."""
        self.status_label.config(text=message, fg=color)
        self.root.update()
    
    def capture_sample(self, hand_landmarks) -> bool:
        """
        Normalize landmarks and save sample.
        
        Returns:
            bool: True if capture succeeded, False otherwise
        """
        # Normalize landmarks
        result = normalize_landmarks(hand_landmarks, threshold=0.05)
        
        if result is None or not result['valid']:
            return False
        
        # Get timestamp
        timestamp = datetime.now().isoformat()
        
        # Create sample record
        sample = {
            'timestamp': timestamp,
            'gesture_id': self.current_gesture_id,
            'gesture_name': self.current_gesture_name,
            'raw_scale_wrist_mcp': result['scale_wrist_mcp'],
            'raw_scale_palm_width': result['scale_palm_width'],
            'scale_used': result['scale_used'],
        }
        
        # Add normalized coordinates
        normalized = result['normalized']
        for i in range(21):
            sample[f'x{i}'] = normalized[i * 2]
            sample[f'y{i}'] = normalized[i * 2 + 1]
        
        # Add to samples list
        self.samples.append(sample)
        self.sample_counts[self.current_gesture_name] += 1
        
        return True
    
    def draw_overlay(self, frame, hand_landmarks, results):
        """Draw landmarks, status, and information overlay on frame."""
        height, width, _ = frame.shape
        
        # Draw hand landmarks
        if hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
        
        # Determine border color based on status
        current_time = time.time()
        border_color = (255, 255, 255)  # Default white
        
        if self.auto_capture_active:
            border_color = (0, 255, 255)  # Yellow for auto-capture
        elif current_time - self.status_time < self.status_duration:
            if self.last_capture_status == "success":
                border_color = (0, 255, 0)  # Green for success
            elif self.last_capture_status == "failed":
                border_color = (0, 0, 255)  # Red for failure
        
        # Draw border
        border_thickness = 15
        cv2.rectangle(frame, (0, 0), (width, height), border_color, border_thickness)
        
        # Draw info panel at top
        panel_height = 180
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Text information
        y_offset = 35
        line_height = 30
        
        # Current gesture
        cv2.putText(
            frame,
            f"Collecting: {self.current_gesture_name}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )
        
        # Sample count
        y_offset += line_height
        current_gesture_total = self.existing_counts[self.current_gesture_name] + self.sample_counts[self.current_gesture_name]
        session_total = sum(self.sample_counts.values())
        overall_total = sum(self.existing_counts.values()) + session_total
        cv2.putText(
            frame,
            f"Current Gesture: {current_gesture_total} | Session: {session_total} | Total: {overall_total}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Scale diagnostics (if hand detected)
        if hand_landmarks:
            result = normalize_landmarks(hand_landmarks, threshold=0.05)
            if result:
                y_offset += line_height
                scale_color = (0, 255, 0) if result['valid'] else (0, 0, 255)
                cv2.putText(
                    frame,
                    f"Scale: {result['scale_used']:.4f} (W-MCP: {result['scale_wrist_mcp']:.4f}, Palm: {result['scale_palm_width']:.4f})",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    scale_color,
                    2
                )
                
                y_offset += line_height
                status_text = "Valid" if result['valid'] else "SKIPPED (scale too small)"
                cv2.putText(
                    frame,
                    f"Status: {status_text}",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    scale_color,
                    2
                )
        
        # Mode indicator
        y_offset += line_height
        mode_text = "AUTO-CAPTURE ACTIVE" if self.auto_capture_active else "Manual Mode"
        mode_color = (0, 255, 255) if self.auto_capture_active else (200, 200, 200)
        cv2.putText(
            frame,
            mode_text,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            mode_color,
            2
        )
        
        return frame
    
    def save_data(self):
        """Append collected samples to the main CSV file."""
        if not self.samples:
            print("No samples to save.")
            return
        
        # Create DataFrame from new samples
        new_df = pd.DataFrame(self.samples)
        
        # Path to main CSV file
        data_dir = Path(__file__).parent.parent / "data"
        main_csv = data_dir / "gestures_data.csv"
        
        # Append to existing file or create new one
        if main_csv.exists():
            # Append without header
            new_df.to_csv(main_csv, mode='a', header=False, index=False)
            print(f"\n{'='*60}")
            print(f"Data appended to: {main_csv}")
        else:
            # Create new file with header
            new_df.to_csv(main_csv, index=False)
            print(f"\n{'='*60}")
            print(f"New data file created: {main_csv}")
        
        print(f"Samples added this session: {len(self.samples)}")
        print(f"\nBreakdown:")
        for name, count in self.sample_counts.items():
            if count > 0:
                print(f"  {name}: {count}")
        
        # Show total count in file
        total_df = pd.read_csv(main_csv)
        print(f"\nTotal samples in file: {len(total_df)}")
        print(f"{'='*60}")
    
    def quit_application(self):
        """Clean up and quit."""
        self.running = False
        self.save_data()
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Main loop."""
        # Start video processing in separate thread
        video_thread = threading.Thread(target=self.video_loop, daemon=True)
        video_thread.start()
        
        # Run tkinter main loop
        self.root.mainloop()
        
        # Cleanup
        self.running = False
        video_thread.join(timeout=1.0)
        self.cap.release()
        cv2.destroyAllWindows()
    
    def video_loop(self):
        """Process video frames in a loop."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Get hand landmarks (only first hand if multiple detected)
            hand_landmarks = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
            
            # Auto-capture logic
            current_time = time.time()
            if self.auto_capture_active and hand_landmarks:
                if current_time - self.last_capture_time >= self.auto_capture_interval:
                    success = self.capture_sample(hand_landmarks)
                    self.last_capture_time = current_time
                    
                    if success:
                        self.last_capture_status = "success"
                        self.update_counts_display()
                    else:
                        self.last_capture_status = "failed"
                    
                    self.status_time = current_time
            
            # Draw overlay
            frame = self.draw_overlay(frame, hand_landmarks, results)
            
            # Show frame
            cv2.imshow('Gesture Data Collector', frame)
            
            # Position window to the right of control panel
            cv2.moveWindow('Gesture Data Collector', 450, 50)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                self.quit_application()
                break
            elif key == ord(' '):  # SPACE - toggle auto-capture
                self.auto_capture_active = not self.auto_capture_active
                if self.auto_capture_active:
                    self.last_capture_time = current_time
                    self.update_status("Auto-capture STARTED", "green")
                else:
                    self.update_status("Auto-capture STOPPED", "orange")
            elif key == 13:  # ENTER - single capture
                if hand_landmarks:
                    success = self.capture_sample(hand_landmarks)
                    if success:
                        self.last_capture_status = "success"
                        self.status_time = current_time
                        self.update_counts_display()
                        self.update_status(f"Captured! ({self.current_gesture_name})", "green")
                    else:
                        self.last_capture_status = "failed"
                        self.status_time = current_time
                        self.update_status("Failed: scale too small", "red")
                else:
                    self.update_status("No hand detected", "red")


def main():
    print("Starting Gesture Data Collector...")
    print("Make sure your webcam is connected and working.")
    print("\nControls:")
    print("  SPACE: Toggle auto-capture mode")
    print("  ENTER: Capture single sample")
    print("  Q or ESC: Save and quit")
    print("\nUse the dropdown in the control window to select gestures.")
    print("="*60)
    
    collector = GestureDataCollector()
    collector.run()


if __name__ == "__main__":
    main()
