# Hand Gesture Recognition - Data Collection

This project implements a real-time hand gesture recognition system with a focus on proper data collection, normalization, and machine learning best practices.

## Project Structure

```
Gesture Recognition/
├── data/
│   ├── gesture_map.json          # Gesture ID to name mapping
│   ├── raw/                       # Raw collected data (CSV files)
│   └── processed/                 # Processed training/test data
├── models/                        # Trained models
├── src/
│   ├── collect_data.py           # Data collection UI
│   ├── normalize.py              # Normalization functions
│   └── main.py                   # Original test script
└── requirements.txt              # Python dependencies
```

## Setup

1. **Create and activate virtual environment** (if not already done):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

## Data Collection

### Running the Collector

```powershell
python src/collect_data.py
```

### Controls

- **SPACE**: Start/Stop auto-capture mode (captures every 0.2s)
- **ENTER**: Capture a single sample
- **Q or ESC**: Save data and quit
- **Dropdown menu**: Select the current gesture to collect

### Features

1. **Real-time visualization**: See hand landmarks, scale diagnostics, and capture status
2. **Color-coded feedback**:
   - White border: Normal mode
   - Yellow border: Auto-capture active
   - Green border: Sample captured successfully
   - Red border: Capture failed (scale too small or invalid)
3. **Scale diagnostics**: Monitor normalization scale factors in real-time
4. **Sample counters**: Track how many samples collected per gesture

### Data Collection Tips

1. **Start with the "null" class**: Collect 100-150 samples of your hand in relaxed, natural poses
2. **Vary your hand position**: Move your hand around the frame while keeping the gesture
3. **Vary rotation**: Rotate your hand slightly (unless the gesture requires specific orientation)
4. **Use auto-capture**: Press SPACE and slowly move your hand to capture natural variations
5. **Check the scale**: Make sure scale values stay above 0.05 (watch the display)
6. **Aim for balance**: Collect similar numbers of samples for each gesture (50-100 each)

### Gesture Definitions

Default gestures (edit `data/gesture_map.json` to customize):

- **null**: No gesture / relaxed hand
- **pinch**: Thumb and index finger touching, fingers facing camera
- **fist**: Closed fist
- **open_palm**: Open hand, fingers spread
- **point**: Index finger extended, others closed
- **thumbs_up**: Thumb extended upward
- **pinch_sideways**: Pinch with fingers pointing sideways

## Normalization Strategy

The system implements **translation and scale invariance**:

### Translation Invariance
- All landmarks are centered by subtracting the wrist position
- This makes features independent of where the hand appears in frame

### Scale Invariance (Hybrid Approach)
Two scale measures are computed:
1. **Wrist-to-middle-MCP distance**: Stable for most gestures
2. **Palm width** (index-MCP to pinky-MCP): Robust to forward/backward hand rotation

The system uses the **maximum** of these two scales, ensuring robust normalization even when one scale collapses (e.g., during wrist-flick gestures).

### Safety Guard
If both scales fall below a threshold (0.05), the frame is rejected to prevent numerical instability.

## Output Format

Collected data is saved as CSV files in `data/raw/` with timestamp names (e.g., `gestures_20251021_143022.csv`).

### CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | string | ISO format timestamp |
| `gesture_id` | int | Numeric gesture label (0-6) |
| `gesture_name` | string | Human-readable gesture name |
| `raw_scale_wrist_mcp` | float | Wrist-to-middle-MCP distance |
| `raw_scale_palm_width` | float | Index-to-pinky MCP distance |
| `scale_used` | float | Actual scale factor used (max of above) |
| `x0, y0, ..., x20, y20` | float × 42 | Normalized landmark coordinates |

## Next Steps (Module 2)

After collecting data:
1. Load and inspect the CSV files
2. Split data into training and test sets
3. Train a classifier (Random Forest or k-NN)
4. Evaluate using confusion matrix
5. Integrate trained model into real-time application

## Troubleshooting

**"Scale too small" messages appearing frequently**:
- This happens when your hand is positioned so wrist and middle finger align (e.g., wrist-flick)
- The hybrid scale approach minimizes this, but extreme poses may still trigger it
- Try slight rotation or repositioning

**Hand not detected**:
- Ensure good lighting
- Keep hand in frame and at reasonable distance
- Avoid backgrounds with skin-like colors
- Check MediaPipe confidence thresholds in `collect_data.py`

**Window not responding**:
- The control window and video window are separate
- Make sure to interact with the correct window for keyboard input
