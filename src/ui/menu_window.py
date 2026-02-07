"""
Menu Window - Main launcher for selecting operational mode.
"""

import tkinter as tk
from tkinter import ttk


class MenuWindow:
    """
    Main menu for selecting gesture recognition mode.

    Provides buttons to launch:
    - Testing Mode: Full display without action execution
    - Compact Mode: Small window with action execution
    - Quit: Exit application
    """

    def __init__(self, config):
        """
        Initialize menu window.

        Args:
            config: AppConfig instance
        """
        self.config = config
        self.root = tk.Tk()
        self.root.title("Hand Gesture Recognition")
        self.root.geometry("450x350")
        self.root.resizable(False, False)

        self._create_widgets()

    def _create_widgets(self):
        """Create GUI widgets."""
        # Title
        title_label = tk.Label(
            self.root,
            text="Hand Gesture Recognition",
            font=("Arial", 18, "bold"),
            pady=20
        )
        title_label.pack()

        # Subtitle
        subtitle_label = tk.Label(
            self.root,
            text="Select Operational Mode",
            font=("Arial", 12),
            fg="gray"
        )
        subtitle_label.pack()

        # Button frame
        button_frame = tk.Frame(self.root, pady=30)
        button_frame.pack()

        # Testing Mode button
        testing_frame = tk.Frame(button_frame)
        testing_frame.pack(pady=10)

        testing_button = tk.Button(
            testing_frame,
            text="Testing Mode",
            command=self.launch_testing_mode,
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            width=20,
            height=2,
            cursor="hand2"
        )
        testing_button.pack()

        testing_desc = tk.Label(
            testing_frame,
            text="Full display • No action execution • Accuracy testing",
            font=("Arial", 9),
            fg="gray"
        )
        testing_desc.pack()

        # Compact Mode button
        compact_frame = tk.Frame(button_frame)
        compact_frame.pack(pady=10)

        compact_button = tk.Button(
            compact_frame,
            text="Compact Mode",
            command=self.launch_compact_mode,
            font=("Arial", 14, "bold"),
            bg="#2196F3",
            fg="white",
            width=20,
            height=2,
            cursor="hand2"
        )
        compact_button.pack()

        compact_desc = tk.Label(
            compact_frame,
            text="Small window • Always-on-top • Executes actions",
            font=("Arial", 9),
            fg="gray"
        )
        compact_desc.pack()

        # Quit button
        quit_button = tk.Button(
            button_frame,
            text="Quit",
            command=self.quit_application,
            font=("Arial", 12),
            bg="#f44336",
            fg="white",
            width=20,
            height=1,
            cursor="hand2"
        )
        quit_button.pack(pady=20)

    def launch_testing_mode(self):
        """Launch testing mode (full display, no actions)."""
        print("Launching Testing Mode...")
        self.root.destroy()  # Close menu

        # Import here to avoid circular dependency
        from src.ui.testing_mode_window import TestingModeWindow

        # Create and run testing mode
        testing_window = TestingModeWindow(self.config)
        testing_window.run()

    def launch_compact_mode(self):
        """Launch compact mode (small window, with actions)."""
        print("Launching Compact Mode...")
        self.root.destroy()  # Close menu

        # Import here to avoid circular dependency
        from src.ui.compact_mode_window import CompactModeWindow

        # Create and run compact mode
        compact_window = CompactModeWindow(self.config)
        compact_window.run()

    def quit_application(self):
        """Exit the application."""
        print("Goodbye!")
        self.root.destroy()

    def run(self):
        """Start the menu window main loop."""
        self.root.mainloop()
