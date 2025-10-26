"""
Model Analysis and Visualization Script

This script provides additional insights into the trained models:
- Feature importance plots
- Model comparison
- Data distribution visualization
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_model(model_path):
    """Load a saved model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def plot_feature_importance(model, save_path):
    """
    Plot feature importance for Random Forest.
    Shows which landmarks are most important for classification.
    """
    importances = model.feature_importances_
    
    # Group by landmark (x and y together)
    landmark_importances = []
    for i in range(21):
        x_importance = importances[i * 2]
        y_importance = importances[i * 2 + 1]
        total_importance = x_importance + y_importance
        landmark_importances.append((i, total_importance, x_importance, y_importance))
    
    # Sort by total importance
    landmark_importances.sort(key=lambda x: x[1], reverse=True)
    
    # Plot top 15 landmarks
    top_n = 15
    landmarks = [f"L{lm[0]}" for lm in landmark_importances[:top_n]]
    x_imp = [lm[2] for lm in landmark_importances[:top_n]]
    y_imp = [lm[3] for lm in landmark_importances[:top_n]]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(landmarks))
    
    ax.bar(x_pos, x_imp, 0.4, label='X coordinate', color='skyblue')
    ax.bar(x_pos + 0.4, y_imp, 0.4, label='Y coordinate', color='lightcoral')
    
    ax.set_xlabel('Landmark ID', fontsize=12)
    ax.set_ylabel('Importance', fontsize=12)
    ax.set_title('Top 15 Most Important Landmarks (Random Forest)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + 0.2)
    ax.set_xticklabels(landmarks, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Feature importance plot saved to: {save_path}")
    plt.close()
    
    # Print landmark names for reference
    landmark_names = [
        "Wrist", "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb Tip",
        "Index MCP", "Index PIP", "Index DIP", "Index Tip",
        "Middle MCP", "Middle PIP", "Middle DIP", "Middle Tip",
        "Ring MCP", "Ring PIP", "Ring DIP", "Ring Tip",
        "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky Tip"
    ]
    
    print("\nTop 10 Most Important Landmarks:")
    for i, (lm_id, total_imp, x_imp, y_imp) in enumerate(landmark_importances[:10], 1):
        print(f"  {i:2}. Landmark {lm_id:2} ({landmark_names[lm_id]:15}): {total_imp:.4f} (x:{x_imp:.4f}, y:{y_imp:.4f})")


def analyze_data_distribution():
    """Analyze and visualize the data distribution."""
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw"
    
    # Load all data
    csv_files = list(raw_dir.glob("gestures_*.csv"))
    all_data = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(all_data, ignore_index=True)
    
    # Plot sample distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    gesture_counts = combined['gesture_id'].value_counts().sort_index()
    gesture_names = ['null', 'pinch', 'fist', 'open_palm', 'point', 'thumbs_up']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    ax1.bar(range(len(gesture_counts)), gesture_counts.values, color=colors[:len(gesture_counts)])
    ax1.set_xlabel('Gesture', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Dataset Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(gesture_counts)))
    ax1.set_xticklabels(gesture_names[:len(gesture_counts)], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(gesture_counts.values):
        ax1.text(i, count + 2, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(gesture_counts.values, labels=gesture_names[:len(gesture_counts)], autopct='%1.1f%%',
            colors=colors[:len(gesture_counts)], startangle=90)
    ax2.set_title('Dataset Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = project_root / "models" / "data_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Data distribution plot saved to: {save_path}")
    plt.close()


def main():
    """Run all analyses."""
    print("="*60)
    print("MODEL ANALYSIS & VISUALIZATION")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    # Load Random Forest model
    print("\n1. Loading Random Forest model...")
    rf_model = load_model(models_dir / "gesture_classifier_latest.pkl")
    print("   ✓ Model loaded")
    
    # Plot feature importance
    print("\n2. Generating feature importance plot...")
    plot_feature_importance(rf_model, models_dir / "feature_importance.png")
    
    # Analyze data distribution
    print("\n3. Analyzing data distribution...")
    analyze_data_distribution()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - feature_importance.png")
    print("  - data_distribution.png")


if __name__ == "__main__":
    main()
