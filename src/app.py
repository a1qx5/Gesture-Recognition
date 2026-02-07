"""
Hand Gesture Recognition Application - Main Entry Point

This application provides two operational modes:
1. Testing Mode: Full display for accuracy testing without action execution
2. Compact Mode: Small always-on-top window with gesture-triggered actions
"""

from src.core.config import AppConfig
from src.ui.menu_window import MenuWindow


def main():
    """Main entry point for the hand gesture recognition application."""
    print("\n" + "="*60)
    print("Hand Gesture Recognition Application")
    print("Powered by MediaPipe & Random Forest")
    print("="*60 + "\n")

    try:
        # Load configuration
        config = AppConfig()

        # Create and run menu
        menu = MenuWindow(config)
        menu.run()

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("  1. Model file exists: models/gesture_classifier_latest.pkl")
        print("  2. Gesture map exists: data/gesture_map.json")
        print("\nRun 'python src/train_model.py' to train a model if needed.")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("Application terminated.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
