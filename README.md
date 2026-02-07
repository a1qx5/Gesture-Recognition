# Hand Gesture Recognition System

Real-time hand gesture recognition application for human-computer interaction via webcam. Control your computer using hand gestures!

## Overview

This application uses MediaPipe for hand detection, scikit-learn for gesture classification, and PyAutoGUI for system-level action execution. It provides two operational modes:

1. **Testing Mode** - Full-screen display for accuracy testing without action execution
2. **Compact Mode** - Small always-on-top window with gesture-triggered actions

## Features

- 🤚 **Real-time hand detection** using MediaPipe (21 landmarks, 30 FPS)
- 🧠 **Machine learning classification** with Random Forest (87% accuracy)
- ⏱️ **Temporal smoothing** to reduce prediction flicker
- 🎯 **Dwell-based triggering** to prevent false positives
- 🖱️ **System-level actions** via PyAutoGUI (clicks, scrolls, etc.)
- 📊 **Comprehensive UI** with diagnostics and feedback
- 🪟 **Always-on-top mode** for unobtrusive operation
- 🔧 **Extensible architecture** for easy customization

## Quick Start

### Installation

```bash
# Navigate to project directory
cd "C:\Users\Alex\Desktop\facultate\Gesture Recognition"

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
python -m src.app
```

This opens a menu with mode selection:
- **Testing Mode** - Test gesture recognition (no actions executed)
- **Compact Mode** - Active gesture control (actions executed)

## Supported Gestures

| Gesture | Description | Default Action |
|---------|-------------|----------------|
| **null** | Relaxed hand | None |
| **pinch** | Thumb + index touching | Left mouse click |
| **fist** | Closed fist | Available (customize) |
| **open_palm** | Flat hand, extended | Available (customize) |
| **point** | Index finger extended | Available (customize) |
| **thumbs_up** | Thumb up | Available (customize) |

## Project Structure (Restructured)

```
src/
├── app.py                      # Main entry point with menu
├── core/                       # Core domain logic
│   ├── gesture_detector.py     # MediaPipe hand detection
│   ├── gesture_recognizer.py   # ML prediction & smoothing
│   ├── action_executor.py      # Gesture-action triggering
│   └── config.py               # Centralized configuration
├── ui/                         # User interface
│   ├── menu_window.py          # Mode selection menu
│   ├── testing_mode_window.py  # Testing mode (no actions)
│   ├── compact_mode_window.py  # Compact mode (with actions)
│   └── ui_utils.py             # Shared UI utilities
├── utils/                      # Utility functions
│   ├── normalize.py            # Landmark normalization
│   └── fps_counter.py          # FPS calculation
├── collect_data.py             # Data collection tool
├── train_model.py              # Model training
└── analyze_model.py            # Model analysis

data/
├── gesture_map.json            # Gesture ID → name mapping
├── gestures_data.csv           # Training data (666 samples)
└── raw/                        # Raw collection sessions

models/
├── gesture_classifier_latest.pkl  # Trained model
└── confusion_matrix_*.png         # Evaluation visuals

docs/
├── ARCHITECTURE.md             # System architecture
├── PIPELINE.md                 # Detailed workflows
└── USAGE.md                    # User guide
```

## Documentation

Comprehensive documentation available in the `docs/` directory:

- **[USAGE.md](docs/USAGE.md)** - Complete user guide with instructions, troubleshooting, and FAQ
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture, component design, extensibility
- **[PIPELINE.md](docs/PIPELINE.md)** - Deep technical details of data flow and processing pipelines

## Data Collection

### Running the Collector

```bash
python src/collect_data.py
```

### Controls

- **SPACE**: Start/Stop auto-capture mode (captures every 0.2s)
- **ENTER**: Capture a single sample
- **Q or ESC**: Save data and quit
- **Dropdown menu**: Select gesture to collect

### Features

- Real-time visualization with hand landmarks
- Color-coded feedback (white/yellow/green/red borders)
- Scale diagnostics display
- Sample counters per gesture

### Data Collection Tips

1. **Start with "null"**: Collect 100-150 samples of relaxed hand
2. **Vary position**: Move hand around frame while maintaining gesture
3. **Vary rotation**: Rotate hand slightly (unless gesture requires specific orientation)
4. **Use auto-capture**: Press SPACE and slowly move hand
5. **Check scale**: Ensure scale > 0.05 (watch display)
6. **Aim for balance**: 50-100 samples per gesture

### CSV Output Format

Saved to `data/gestures_data.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | string | ISO timestamp |
| `gesture_id` | int | Gesture label (0-5) |
| `gesture_name` | string | Readable name |
| `raw_scale_wrist_mcp` | float | Wrist-to-middle-MCP distance |
| `raw_scale_palm_width` | float | Palm width |
| `scale_used` | float | Actual scale (max of above) |
| `x0, y0, ..., x20, y20` | float × 42 | Normalized coordinates |

## Normalization Strategy

### Translation Invariance
- All landmarks centered by subtracting wrist position
- Features independent of hand location in frame

### Scale Invariance (Hybrid Approach)
Two scale measures computed:
1. **Wrist-to-middle-MCP**: Stable for most gestures
2. **Palm width** (index-MCP to pinky-MCP): Robust to rotation

System uses **maximum** of these scales for robust normalization.

### Safety Guard
Frames rejected if scale < 0.05 to prevent numerical instability.

## Training the Model

```bash
python src/train_model.py
```

Trains a Random Forest classifier on collected data:
- 80/20 train-test split
- 5-fold cross-validation
- Generates confusion matrices
- Saves to `models/gesture_classifier_latest.pkl`

## Customization

### Adding Gesture-Action Mappings

Edit `src/core/config.py`:

```python
GESTURE_ACTIONS = {
    "pinch": "left_click",
    "fist": "right_click",      # Add this
    "thumbs_up": "scroll_up",   # Add this
}
```

Restart application to apply changes.

### Training New Gestures

1. Add gesture to `data/gesture_map.json`
2. Collect samples: `python src/collect_data.py`
3. Train model: `python src/train_model.py`
4. Map to action in `config.py`

See [USAGE.md](docs/USAGE.md) for detailed instructions.

## Key Technologies

- **MediaPipe Hands** - Hand landmark detection (21 points)
- **OpenCV** - Webcam capture and video display
- **scikit-learn** - Random Forest classifier (100 trees)
- **NumPy** - Numerical computations
- **PyAutoGUI** - System-level action execution
- **Tkinter** - GUI menu

## System Requirements

**Minimum**:
- Python 3.8+
- CPU: Intel i3 or equivalent
- RAM: 2 GB
- Camera: 720p webcam

**Recommended**:
- Python 3.10+
- CPU: Intel i5 or equivalent
- RAM: 4 GB
- Camera: 1080p webcam

## Performance

- **FPS**: 25-30 frames per second
- **Latency**: ~35ms per frame
- **Memory**: ~80 MB
- **CPU**: 15-30% (single core)
- **Model inference**: ~3ms per prediction

## Pipeline Overview

```
Frame Capture → Hand Detection → Landmark Normalization →
ML Prediction → Temporal Smoothing → Action Triggering →
UI Rendering → Display
```

See [PIPELINE.md](docs/PIPELINE.md) for frame-by-frame technical details.

## Troubleshooting

### Common Issues

**"Model not found" error**:
```bash
python src/train_model.py
```

**Low FPS (<20)**:
- Close other camera applications
- Reduce resolution in `src/core/config.py`

**Gesture not recognized**:
- Check hand distance (1-2 feet from camera)
- Improve lighting
- Collect more training samples (100+ per gesture)

**Actions not triggering** (Compact Mode):
- Ensure you're in Compact Mode (not Testing Mode)
- Check gesture mapping in `config.py`
- Hold gesture steady for ~0.15 seconds
- Verify console shows "✓ ACTION executed"

**Always-on-top not working**:
- Update OpenCV: `pip install --upgrade opencv-python`
- On Windows: `pip install pywin32` (optional)

**"Scale too small" messages**:
- Move hand to 1-2 feet distance
- Avoid extreme hand orientations
- Slight rotation may help

**Hand not detected**:
- Ensure good lighting
- Keep hand clearly in frame
- Avoid skin-colored backgrounds
- Check MediaPipe confidence thresholds

See [USAGE.md](docs/USAGE.md) for complete troubleshooting guide.

## Architecture Highlights

- **Separation of Concerns**: Core logic, UI, and utilities separated
- **Single Responsibility**: Each class has one clear purpose
- **Dependency Injection**: Components receive dependencies via constructor
- **Extensible Design**: Easy to add new gestures and actions
- **Well-Documented**: Comprehensive docstrings and documentation

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design documentation.

## Development

### Adding Custom Actions

Edit `src/core/action_executor.py`:

```python
def _execute_screenshot(self):
    """Take a screenshot."""
    pyautogui.screenshot('screenshot.png')
    self.last_action = "SCREENSHOT"
    self.action_display_frames = 30
```

Register handler and map in `config.py`. See documentation for details.

### Testing

```bash
# Test gesture recognition (no actions)
python -m src.app  # Select "Testing Mode"

# Test gesture control (with actions)
python -m src.app  # Select "Compact Mode"

# Test data collection
python src/collect_data.py

# Test model training
python src/train_model.py
```

## Dataset

- **Total samples**: 666
- **Gestures**: 6 classes
- **Features**: 42 (21 landmarks × 2 coordinates)
- **Normalization**: Translation and scale invariant
- **Format**: CSV with metadata and normalized coordinates

## Model Details

- **Algorithm**: Random Forest (100 trees)
- **Features**: 42 normalized landmark coordinates
- **Train accuracy**: ~98%
- **Test accuracy**: ~95%
- **Inference time**: ~3ms per frame

## Safety & Privacy

- **No recording**: Video processed in real-time only
- **No network**: All processing happens locally
- **No storage**: Frames not saved to disk
- **Dwell-based triggering**: Minimizes accidental actions

## Future Enhancements

- Multi-hand support
- Gesture sequences (swipe patterns)
- Deep learning model (CNN/RNN)
- System tray integration
- Gesture customization UI
- Confidence thresholding
- Cross-platform optimization

## Acknowledgments

- **MediaPipe** by Google for hand landmark detection
- **scikit-learn** for machine learning algorithms
- **OpenCV** for computer vision utilities
- **PyAutoGUI** for system automation

---

**Built for human-computer interaction research** 🤚✨
