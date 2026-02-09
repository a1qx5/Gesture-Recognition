"""
Model Training Script for Hand Gesture Recognition

This script implements Module 2: Model Training & Evaluation
- Loads and merges all collected data
- Splits into train/test sets (80/20)
- Trains Random Forest and k-NN classifiers
- Evaluates with accuracy metrics and confusion matrices
- Saves the best model for inference
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class GestureModelTrainer:
    def __init__(self):
        """Initialize the trainer with paths and configurations."""
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = self.project_root / "models"
        
        # Load gesture map
        with open(self.data_dir / "gesture_map.json", 'r') as f:
            self.gesture_map = json.load(f)
        
        self.gesture_names = [self.gesture_map[str(i)] for i in sorted([int(k) for k in self.gesture_map.keys()])]
        
        # Data containers
        self.X = None  # Features (normalized coordinates)
        self.y = None  # Labels (gesture IDs)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Models
        self.rf_model = None
        self.knn_model = None
        
    def load_and_merge_data(self):
        """
        Load data from the main gestures_data.csv file.
        
        Returns:
            pd.DataFrame: Dataset
        """
        print("\n" + "="*60)
        print("STEP 1: Loading Data")
        print("="*60)
        
        main_csv = self.data_dir / "gestures_data.csv"
        
        if not main_csv.exists():
            raise FileNotFoundError(f"Main data file not found: {main_csv}")
        
        print(f"Loading from: {main_csv.name}")
        
        # Load the main CSV file
        df = pd.read_csv(main_csv)
        
        print(f"\nTotal samples loaded: {len(df)}")
        print(f"\nSamples per gesture:")
        gesture_counts = df['gesture_id'].value_counts().sort_index()
        for gesture_id, count in gesture_counts.items():
            gesture_name = self.gesture_map.get(str(gesture_id), "Unknown")
            print(f"  {gesture_name:15} (ID {gesture_id}): {count:4} samples")
        
        return df
    
    def prepare_features_and_labels(self, df):
        """
        Extract feature matrix X (normalized coordinates) and label vector y (gesture IDs).
        
        Args:
            df: Combined DataFrame
            
        Sets:
            self.X: Feature matrix (n_samples, 42 features)
            self.y: Label vector (n_samples,)
        """
        print("\n" + "="*60)
        print("STEP 2: Preparing Features and Labels")
        print("="*60)
        
        # Extract coordinate columns (x0, y0, x1, y1, ..., x20, y20)
        coordinate_columns = []
        for i in range(21):  # 21 landmarks
            coordinate_columns.append(f'x{i}')
            coordinate_columns.append(f'y{i}')
        
        # Feature matrix X
        self.X = df[coordinate_columns].values
        
        # Label vector y
        self.y = df['gesture_id'].values
        
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"  - {self.X.shape[0]} samples")
        print(f"  - {self.X.shape[1]} features (21 landmarks × 2 coordinates)")
        print(f"\nLabel vector shape: {self.y.shape}")
        print(f"  - {self.y.shape[0]} labels")
        print(f"  - {len(np.unique(self.y))} unique classes")
        
        # Check for any NaN or Inf values
        if np.any(np.isnan(self.X)) or np.any(np.isinf(self.X)):
            print("\n⚠️  WARNING: Found NaN or Inf values in features!")
            print(f"   NaN count: {np.sum(np.isnan(self.X))}")
            print(f"   Inf count: {np.sum(np.isinf(self.X))}")
            print("   Removing affected samples...")
            
            valid_mask = np.all(np.isfinite(self.X), axis=1)
            self.X = self.X[valid_mask]
            self.y = self.y[valid_mask]
            
            print(f"   Samples after cleaning: {len(self.X)}")
        else:
            print("\n✓ No NaN or Inf values detected")
    
    def split_train_test(self, test_size=0.2, random_state=42):
        """
        Split data into training and test sets with stratification.
        
        The Golden Rule: Never test on data you trained on!
        - Stratify ensures both sets have the same class proportions
        - Random state ensures reproducibility
        
        Args:
            test_size: Proportion of data for testing (default 0.2 = 20%)
            random_state: Random seed for reproducibility
        """
        print("\n" + "="*60)
        print("STEP 3: Train-Test Split")
        print("="*60)
        
        print(f"Split ratio: {int((1-test_size)*100)}/{int(test_size*100)} (train/test)")
        print(f"Using stratification to maintain class balance")
        print(f"Random state: {random_state} (for reproducibility)")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y  # Critical: maintains class proportions
        )
        
        print(f"\nTraining set: {len(self.X_train)} samples")
        print(f"Test set:     {len(self.X_test)} samples")
        
        print(f"\nTraining set class distribution:")
        train_counts = pd.Series(self.y_train).value_counts().sort_index()
        for gesture_id, count in train_counts.items():
            gesture_name = self.gesture_map.get(str(gesture_id), "Unknown")
            percentage = (count / len(self.y_train)) * 100
            print(f"  {gesture_name:15}: {count:3} ({percentage:5.1f}%)")
        
        print(f"\nTest set class distribution:")
        test_counts = pd.Series(self.y_test).value_counts().sort_index()
        for gesture_id, count in test_counts.items():
            gesture_name = self.gesture_map.get(str(gesture_id), "Unknown")
            percentage = (count / len(self.y_test)) * 100
            print(f"  {gesture_name:15}: {count:3} ({percentage:5.1f}%)")
    
    def train_random_forest(self, n_estimators=100, max_depth=None, min_samples_split=2):
        """
        Train a Random Forest classifier.
        
        Random Forest Intuition:
        - Ensemble of decision trees (like a committee of experts)
        - Each tree sees a random subset of data (bootstrap sampling)
        - Each split considers a random subset of features
        - Final prediction: majority vote from all trees
        
        Why it works:
        - Individual trees might overfit, but averaging reduces variance
        - Random feature selection makes trees diverse (different mistakes)
        - Resistant to overfitting if enough trees are used
        
        Args:
            n_estimators: Number of trees (more = better, but slower)
            max_depth: Maximum tree depth (None = no limit, lower = more regularization)
            min_samples_split: Min samples to split a node (higher = more regularization)
        """
        print("\n" + "="*60)
        print("STEP 4: Training Random Forest Classifier")
        print("="*60)
        
        print(f"Hyperparameters:")
        print(f"  - n_estimators: {n_estimators} (number of trees)")
        print(f"  - max_depth: {max_depth} (tree depth limit)")
        print(f"  - min_samples_split: {min_samples_split} (split threshold)")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        print(f"\nTraining on {len(self.X_train)} samples...")
        self.rf_model.fit(self.X_train, self.y_train)
        print("✓ Training complete!")
        
        # Evaluate on training set
        y_train_pred = self.rf_model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        
        # Evaluate on test set
        y_test_pred = self.rf_model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        print(f"\nResults:")
        print(f"  Training accuracy: {train_accuracy*100:.2f}%")
        print(f"  Test accuracy:     {test_accuracy*100:.2f}%")
        print(f"  Gap:               {abs(train_accuracy - test_accuracy)*100:.2f}%")
        
        # Interpret the gap
        gap = train_accuracy - test_accuracy
        if gap < 0.05:
            print("  → Good fit! Model generalizes well.")
        elif gap < 0.15:
            print("  → Acceptable fit. Slight overfitting.")
        else:
            print("  → Overfitting detected! Consider regularization or more data.")
        
        # Cross-validation for more robust estimate
        print(f"\n5-Fold Cross-Validation (on training set):")
        cv_scores = cross_val_score(self.rf_model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
        print(f"  Individual folds: {[f'{s*100:.1f}%' for s in cv_scores]}")
        
        # Feature importance
        print(f"\nTop 10 Most Important Features:")
        feature_names = []
        for i in range(21):
            feature_names.append(f'x{i}')
            feature_names.append(f'y{i}')
        
        importances = self.rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        for rank, idx in enumerate(indices, 1):
            landmark_num = idx // 2
            coord = 'x' if idx % 2 == 0 else 'y'
            print(f"  {rank:2}. {feature_names[idx]:6} (Landmark {landmark_num}, {coord}): {importances[idx]:.4f}")
        
        return y_test_pred
    
    def train_knn(self, n_neighbors=5):
        """
        Train a k-Nearest Neighbors classifier.
        
        k-NN Intuition:
        - "You are the average of your k closest friends"
        - Finds k most similar training samples (by Euclidean distance)
        - Predicts the most common class among those k neighbors
        
        Why it works:
        - Simple and interpretable (you can inspect which samples influenced prediction)
        - No assumptions about data distribution
        - Works well with normalized features
        
        Tradeoffs:
        - Slow at inference (must compare to all training samples)
        - Sensitive to k (too small = overfitting, too large = underfitting)
        - Can struggle with high-dimensional data
        
        Args:
            n_neighbors: Number of neighbors to consider (typically 3, 5, or 7)
        """
        print("\n" + "="*60)
        print("STEP 5: Training k-Nearest Neighbors Classifier")
        print("="*60)
        
        print(f"Hyperparameters:")
        print(f"  - n_neighbors: {n_neighbors}")
        print(f"  - metric: euclidean (distance measure)")
        
        self.knn_model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric='euclidean',
            n_jobs=-1
        )
        
        print(f"\nTraining on {len(self.X_train)} samples...")
        self.knn_model.fit(self.X_train, self.y_train)
        print("✓ Training complete! (k-NN is lazy: actual work happens at prediction time)")
        
        # Evaluate on training set
        y_train_pred = self.knn_model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        
        # Evaluate on test set
        y_test_pred = self.knn_model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        print(f"\nResults:")
        print(f"  Training accuracy: {train_accuracy*100:.2f}%")
        print(f"  Test accuracy:     {test_accuracy*100:.2f}%")
        print(f"  Gap:               {abs(train_accuracy - test_accuracy)*100:.2f}%")
        
        # Interpret the gap
        gap = train_accuracy - test_accuracy
        if gap < 0.05:
            print("  → Good fit! Model generalizes well.")
        elif gap < 0.15:
            print("  → Acceptable fit. Slight overfitting.")
        else:
            print("  → Overfitting detected! Try increasing k.")
        
        # Cross-validation
        print(f"\n5-Fold Cross-Validation (on training set):")
        cv_scores = cross_val_score(self.knn_model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
        print(f"  Individual folds: {[f'{s*100:.1f}%' for s in cv_scores]}")
        
        return y_test_pred
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path):
        """
        Generate and display confusion matrix.
        
        Confusion Matrix shows:
        - Diagonal: Correct predictions (good!)
        - Off-diagonal: Mistakes (which gestures are confused)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name for the plot title
            save_path: Where to save the figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Display confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[self.gesture_map[str(i)] for i in sorted([int(k) for k in self.gesture_map.keys()]) if i in np.unique(y_true)]
        )
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved to: {save_path}")
        
        return cm
    
    def analyze_confusion_matrix(self, cm, model_name):
        """
        Analyze confusion matrix to identify which gestures are most confused.
        
        Args:
            cm: Confusion matrix
            model_name: Model name for display
        """
        print(f"\n{model_name} - Confusion Analysis:")
        print("-" * 60)
        
        # Get unique classes in the data
        unique_classes = sorted(np.unique(self.y_test))
        
        # Per-class accuracy
        print("Per-class accuracy:")
        for i, class_id in enumerate(unique_classes):
            gesture_name = self.gesture_map[str(int(class_id))]
            class_total = cm[i].sum()
            class_correct = cm[i, i]
            class_accuracy = (class_correct / class_total * 100) if class_total > 0 else 0
            print(f"  {gesture_name:15}: {class_accuracy:5.1f}% ({class_correct}/{class_total})")

        # Find most confused pairs
        print(f"\nMost confused gesture pairs:")
        confusion_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    true_gesture = self.gesture_map[str(int(unique_classes[i]))]
                    pred_gesture = self.gesture_map[str(int(unique_classes[j]))]
                    confusion_pairs.append((cm[i, j], true_gesture, pred_gesture))
        
        confusion_pairs.sort(reverse=True)
        
        if confusion_pairs:
            for count, true_g, pred_g in confusion_pairs[:5]:  # Top 5
                print(f"  {true_g:15} → {pred_g:15}: {count} times")
        else:
            print("  No confusion! Perfect classification!")
    
    def save_models(self):
        """Save trained models to disk."""
        print("\n" + "="*60)
        print("STEP 6: Saving Models")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save Random Forest
        if self.rf_model:
            rf_path = self.models_dir / f"random_forest_{timestamp}.pkl"
            with open(rf_path, 'wb') as f:
                pickle.dump(self.rf_model, f)
            print(f"✓ Random Forest saved to: {rf_path}")
        
        # Save k-NN
        if self.knn_model:
            knn_path = self.models_dir / f"knn_{timestamp}.pkl"
            with open(knn_path, 'wb') as f:
                pickle.dump(self.knn_model, f)
            print(f"✓ k-NN saved to: {knn_path}")
        
        # Save a copy of the best model as "latest"
        # (You can choose which model to use based on test accuracy)
        best_model_path = self.models_dir / "gesture_classifier_latest.pkl"
        with open(best_model_path, 'wb') as f:
            pickle.dump(self.rf_model, f)  # Using RF as default
        print(f"✓ Latest model saved to: {best_model_path}")
    
    def run_full_pipeline(self):
        """Run the complete training and evaluation pipeline."""
        print("\n" + "="*60)
        print("HAND GESTURE RECOGNITION - MODEL TRAINING PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        df = self.load_and_merge_data()
        
        # Step 2: Prepare features
        self.prepare_features_and_labels(df)
        
        # Step 3: Split train/test
        self.split_train_test(test_size=0.2, random_state=42)
        
        # Step 4: Train Random Forest
        rf_predictions = self.train_random_forest(n_estimators=100, max_depth=None, min_samples_split=2)
        
        # Step 5: Train k-NN
        knn_predictions = self.train_knn(n_neighbors=5)
        
        # Step 6: Confusion matrices
        print("\n" + "="*60)
        print("STEP 6: Generating Confusion Matrices")
        print("="*60)
        
        print("\nRandom Forest:")
        rf_cm = self.plot_confusion_matrix(
            self.y_test,
            rf_predictions,
            "Random Forest",
            self.models_dir / "confusion_matrix_rf.png"
        )
        self.analyze_confusion_matrix(rf_cm, "Random Forest")
        
        print("\nk-Nearest Neighbors:")
        knn_cm = self.plot_confusion_matrix(
            self.y_test,
            knn_predictions,
            "k-Nearest Neighbors",
            self.models_dir / "confusion_matrix_knn.png"
        )
        self.analyze_confusion_matrix(knn_cm, "k-Nearest Neighbors")
        
        # Step 7: Save models
        self.save_models()
        
        # Final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
        rf_acc = accuracy_score(self.y_test, rf_predictions) * 100
        knn_acc = accuracy_score(self.y_test, knn_predictions) * 100
        
        print(f"\nFinal Test Accuracies:")
        print(f"  Random Forest: {rf_acc:.2f}%")
        print(f"  k-NN:          {knn_acc:.2f}%")
        
        if rf_acc > knn_acc:
            print(f"\n✓ Random Forest performed better (+{rf_acc - knn_acc:.2f}%)")
        elif knn_acc > rf_acc:
            print(f"\n✓ k-NN performed better (+{knn_acc - rf_acc:.2f}%)")
        else:
            print(f"\n✓ Both models tied!")
        
        print(f"\nNext steps:")
        print(f"  1. Inspect confusion matrices to see which gestures are confused")
        print(f"  2. If accuracy is low, collect more data for confused gestures")
        print(f"  3. Use the saved model in real-time inference (main.py)")


def main():
    """Main entry point."""
    trainer = GestureModelTrainer()
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()
