"""
Gesture Recognizer - Gesture prediction with temporal smoothing.
"""

import pickle
import json
import numpy as np
from pathlib import Path


class GestureRecognizer:
    """
    Handles gesture prediction using a trained ML model with temporal smoothing.

    Responsibilities:
    - Load trained Random Forest model
    - Load gesture name mappings
    - Normalize hand landmarks (via utils.normalize)
    - Predict gestures from normalized features
    - Apply temporal smoothing to reduce flickering
    """

    def __init__(self, model_path, gesture_map_path, history_size=5):
        """
        Initialize the gesture recognizer.

        Args:
            model_path: Path to trained model (.pkl file)
            gesture_map_path: Path to gesture map JSON file
            history_size: Number of frames to use for temporal smoothing
        """
        # Load gesture map
        with open(gesture_map_path, 'r') as f:
            self.gesture_map = json.load(f)

        # Load trained model
        print(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("✓ Model loaded successfully!")

        # Temporal smoothing state
        self.prediction_history = []
        self.history_size = history_size
        self.last_confidence = 0.0

    def predict(self, hand_landmarks, normalize_func):
        """
        Predict gesture from hand landmarks (single frame, no smoothing).

        Args:
            hand_landmarks: MediaPipe hand landmarks object
            normalize_func: Function to normalize landmarks (from utils.normalize)

        Returns:
            tuple: (gesture_name, confidence, valid)
                - gesture_name: Predicted gesture or status message
                - confidence: Prediction confidence (0-1)
                - valid: Boolean indicating if normalization succeeded
        """
        # Normalize landmarks
        result = normalize_func(hand_landmarks, threshold=0.05)

        if result is None or not result['valid']:
            return "Invalid (scale too small)", 0.0, False

        # Get normalized features
        features = result['normalized'].reshape(1, -1)  # Shape: (1, 42)

        # Predict
        prediction_id = self.model.predict(features)[0]

        # Get prediction probabilities (if model supports it)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities[prediction_id]
        else:
            confidence = 1.0  # k-NN doesn't have probabilities

        # Map prediction ID to gesture name
        gesture_name = self.gesture_map.get(str(prediction_id), "Unknown")

        self.last_confidence = confidence
        return gesture_name, confidence, True

    def predict_smooth(self, hand_landmarks, normalize_func):
        """
        Predict gesture with temporal smoothing (majority voting).

        Args:
            hand_landmarks: MediaPipe hand landmarks object
            normalize_func: Function to normalize landmarks

        Returns:
            tuple: (gesture_name, confidence, valid)
        """
        # Get single-frame prediction
        gesture, confidence, valid = self.predict(hand_landmarks, normalize_func)

        if not valid:
            return gesture, confidence, False

        # Apply temporal smoothing
        smoothed_gesture = self._smooth_prediction(gesture)

        return smoothed_gesture, confidence, True

    def _smooth_prediction(self, gesture_name):
        """
        Apply temporal smoothing using majority voting over recent frames.

        Args:
            gesture_name: Current frame's prediction

        Returns:
            str: Smoothed gesture name (most common in recent history)
        """
        self.prediction_history.append(gesture_name)

        # Keep only last N predictions
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)

        # Return most common prediction
        if self.prediction_history:
            # Count occurrences
            counts = {}
            for pred in self.prediction_history:
                counts[pred] = counts.get(pred, 0) + 1

            # Return most frequent
            return max(counts, key=counts.get)

        return gesture_name

    def get_confidence(self):
        """Return the last confidence score."""
        return self.last_confidence

    def reset_history(self):
        """Clear the prediction history (e.g., when hand disappears)."""
        self.prediction_history = []
