"""
Normalization functions for hand landmark data.
Implements translation and scale invariance as discussed in Module 1.
"""

import numpy as np
from typing import Tuple, Optional, Dict


def get_landmark_array(hand_landmarks) -> np.ndarray:
    """
    Convert MediaPipe hand landmarks to numpy array.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object
        
    Returns:
        np.ndarray: Shape (21, 2) containing x, y coordinates
    """
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append([landmark.x, landmark.y])
    return np.array(landmarks)


def compute_scale_wrist_mcp(landmarks: np.ndarray) -> float:
    """
    Compute scale as distance from wrist (0) to middle finger MCP (9).
    
    Args:
        landmarks: Shape (21, 2) array of x, y coordinates
        
    Returns:
        float: Euclidean distance between wrist and middle MCP
    """
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    distance = np.linalg.norm(middle_mcp - wrist)
    return distance


def compute_scale_palm_width(landmarks: np.ndarray) -> float:
    """
    Compute scale as palm width (distance from index MCP to pinky MCP).
    
    Args:
        landmarks: Shape (21, 2) array of x, y coordinates
        
    Returns:
        float: Euclidean distance between index MCP and pinky MCP
    """
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    distance = np.linalg.norm(pinky_mcp - index_mcp)
    return distance


def normalize_landmarks(
    hand_landmarks,
    threshold: float = 0.05
) -> Optional[Dict[str, any]]:
    """
    Normalize hand landmarks using hybrid scale approach.
      
    Args:
        hand_landmarks: MediaPipe hand landmarks object
        threshold: Minimum acceptable scale value (default 0.05)
        
    Returns:
        Dictionary containing:
            - 'normalized': Shape (42,) flattened normalized coordinates
            - 'scale_wrist_mcp': Wrist-to-middle-MCP distance
            - 'scale_palm_width': Palm width distance
            - 'scale_used': Actual scale factor used
            - 'valid': Boolean indicating if normalization succeeded
        Returns None if normalization fails (scale too small)
    """
    # Convert to numpy array
    landmarks = get_landmark_array(hand_landmarks)
    
    # Step 1: Center by subtracting wrist (translation invariance)
    wrist = landmarks[0]
    centered = landmarks - wrist
    
    # Step 2: Compute two scale measures
    scale_wrist_mcp = compute_scale_wrist_mcp(landmarks)
    scale_palm_width = compute_scale_palm_width(landmarks)
    
    # Step 3: Hybrid scale - use maximum
    scale_used = max(scale_wrist_mcp, scale_palm_width)
    
    # Step 4: Guard against scale collapse
    if scale_used < threshold:
        return {
            'normalized': None,
            'scale_wrist_mcp': scale_wrist_mcp,
            'scale_palm_width': scale_palm_width,
            'scale_used': scale_used,
            'valid': False
        }
    
    # Step 5: Normalize by scale (scale invariance)
    normalized = centered / scale_used
    
    # Step 6: Flatten to feature vector [x0, y0, x1, y1, ..., x20, y20]
    normalized_flat = normalized.flatten()
    
    # Check for NaN or Inf
    if not np.all(np.isfinite(normalized_flat)):
        return {
            'normalized': None,
            'scale_wrist_mcp': scale_wrist_mcp,
            'scale_palm_width': scale_palm_width,
            'scale_used': scale_used,
            'valid': False
        }
    
    return {
        'normalized': normalized_flat,
        'scale_wrist_mcp': scale_wrist_mcp,
        'scale_palm_width': scale_palm_width,
        'scale_used': scale_used,
        'valid': True
    }


def get_average_confidence(hand_landmarks) -> float:
    """
    Compute average confidence score across all landmarks.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object
        
    Returns:
        float: Average confidence score (0-1)
    """
    confidences = []
    for landmark in hand_landmarks.landmark:
        # placeholder
        confidences.append(1.0)
    return np.mean(confidences)
