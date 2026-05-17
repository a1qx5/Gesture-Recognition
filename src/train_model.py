"""
Model Training Script for Hand Gesture Recognition

This script implements Module 2: Model Training & Evaluation
- Loads and merges all collected data
- Splits into train/test sets (80/20)
- Trains Random Forest, k-NN, and SVM classifiers
- Evaluates with accuracy, F1-score, and normalized confusion matrices
- Generates learning curves and a classifier comparison bar chart
- Performs paired t-tests on cross-validation scores
- Saves the best model for inference
"""

import time
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, f1_score
)
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no tkinter required
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
        self.svm_model = None

        # CV fold scores and training times -- stored for statistical comparison
        self.cv_scores = {}
        self.train_times = {}

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

        coordinate_columns = []
        for i in range(21):
            coordinate_columns.append(f'x{i}')
            coordinate_columns.append(f'y{i}')

        self.X = df[coordinate_columns].values
        self.y = df['gesture_id'].values

        print(f"Feature matrix shape: {self.X.shape}")
        print(f"  - {self.X.shape[0]} samples")
        print(f"  - {self.X.shape[1]} features (21 landmarks x 2 coordinates)")
        print(f"\nLabel vector shape: {self.y.shape}")
        print(f"  - {self.y.shape[0]} labels")
        print(f"  - {len(np.unique(self.y))} unique classes")

        if np.any(np.isnan(self.X)) or np.any(np.isinf(self.X)):
            print("\nWARNING: Found NaN or Inf values in features!")
            print(f"   NaN count: {np.sum(np.isnan(self.X))}")
            print(f"   Inf count: {np.sum(np.isinf(self.X))}")
            print("   Removing affected samples...")

            valid_mask = np.all(np.isfinite(self.X), axis=1)
            self.X = self.X[valid_mask]
            self.y = self.y[valid_mask]

            print(f"   Samples after cleaning: {len(self.X)}")
        else:
            print("\nOK No NaN or Inf values detected")

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
            stratify=self.y
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
            n_jobs=-1
        )

        print(f"\nTraining on {len(self.X_train)} samples...")
        t0 = time.perf_counter()
        self.rf_model.fit(self.X_train, self.y_train)
        self.train_times['Random Forest'] = time.perf_counter() - t0
        print(f"OK Training complete! ({self.train_times['Random Forest']:.2f}s)")

        y_train_pred = self.rf_model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)

        y_test_pred = self.rf_model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)

        print(f"\nResults:")
        print(f"  Training accuracy: {train_accuracy*100:.2f}%")
        print(f"  Test accuracy:     {test_accuracy*100:.2f}%")
        print(f"  Gap:               {abs(train_accuracy - test_accuracy)*100:.2f}%")

        gap = train_accuracy - test_accuracy
        if gap < 0.05:
            print("  -> Good fit! Model generalizes well.")
        elif gap < 0.15:
            print("  -> Acceptable fit. Slight overfitting.")
        else:
            print("  -> Overfitting detected! Consider regularization or more data.")

        print(f"\n5-Fold Cross-Validation (on training set):")
        cv_scores = cross_val_score(self.rf_model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        self.cv_scores['Random Forest'] = cv_scores
        print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
        print(f"  Individual folds: {[f'{s*100:.1f}%' for s in cv_scores]}")

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
        t0 = time.perf_counter()
        self.knn_model.fit(self.X_train, self.y_train)
        self.train_times['KNN'] = time.perf_counter() - t0
        print(f"OK Training complete! ({self.train_times['KNN']:.4f}s -- k-NN is lazy: actual work happens at inference)")

        y_train_pred = self.knn_model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)

        y_test_pred = self.knn_model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)

        print(f"\nResults:")
        print(f"  Training accuracy: {train_accuracy*100:.2f}%")
        print(f"  Test accuracy:     {test_accuracy*100:.2f}%")
        print(f"  Gap:               {abs(train_accuracy - test_accuracy)*100:.2f}%")

        gap = train_accuracy - test_accuracy
        if gap < 0.05:
            print("  -> Good fit! Model generalizes well.")
        elif gap < 0.15:
            print("  -> Acceptable fit. Slight overfitting.")
        else:
            print("  -> Overfitting detected! Try increasing k.")

        print(f"\n5-Fold Cross-Validation (on training set):")
        cv_scores = cross_val_score(self.knn_model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        self.cv_scores['KNN'] = cv_scores
        print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
        print(f"  Individual folds: {[f'{s*100:.1f}%' for s in cv_scores]}")

        return y_test_pred

    def train_svm(self, C=10, gamma='scale'):
        """
        Train a Support Vector Machine classifier with an RBF kernel.

        SVM Intuition:
        - Finds the hyperplane that maximally separates classes in feature space
        - The "margin" is the gap between the hyperplane and the nearest samples
        - Support vectors: the training points closest to the decision boundary
        - RBF kernel: implicitly maps features into a higher-dimensional space,
          allowing non-linear decision boundaries without computing them explicitly

        Why it works for gesture recognition:
        - Hand pose landmarks occupy a non-linearly separable space (e.g., "ok"
          and "pinch" differ subtly in finger curvature, not just position)
        - The RBF kernel captures local similarity between poses effectively
        - Margin maximization provides good generalization from limited data

        Tradeoffs:
        - Slower to train than k-NN (quadratic in n_samples for exact solvers)
        - Hyperparameter C controls the margin/error tradeoff:
            high C = smaller margin, fits training data more tightly (risk of overfit)
            low C  = larger margin, more misclassifications allowed (risk of underfit)
        - probability=True adds Platt scaling for confidence estimates (small overhead)

        Args:
            C: Regularization parameter (default 10 -- moderate tightness)
            gamma: RBF kernel width ('scale' = 1 / (n_features * X.var()))
        """
        print("\n" + "="*60)
        print("STEP 6: Training Support Vector Machine Classifier")
        print("="*60)

        print(f"Hyperparameters:")
        print(f"  - kernel: RBF (Radial Basis Function)")
        print(f"  - C: {C} (regularization -- penalty for margin violations)")
        print(f"  - gamma: {gamma} (RBF kernel width)")
        print(f"  - probability: True (enables confidence estimates via Platt scaling)")

        self.svm_model = SVC(
            kernel='rbf',
            C=C,
            gamma=gamma,
            probability=True,
            random_state=42
        )

        print(f"\nTraining on {len(self.X_train)} samples...")
        t0 = time.perf_counter()
        self.svm_model.fit(self.X_train, self.y_train)
        self.train_times['SVM'] = time.perf_counter() - t0
        print(f"OK Training complete! ({self.train_times['SVM']:.2f}s)")

        y_train_pred = self.svm_model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)

        y_test_pred = self.svm_model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)

        print(f"\nResults:")
        print(f"  Training accuracy: {train_accuracy*100:.2f}%")
        print(f"  Test accuracy:     {test_accuracy*100:.2f}%")
        print(f"  Gap:               {abs(train_accuracy - test_accuracy)*100:.2f}%")

        gap = train_accuracy - test_accuracy
        if gap < 0.05:
            print("  -> Good fit! Model generalizes well.")
        elif gap < 0.15:
            print("  -> Acceptable fit. Slight overfitting.")
        else:
            print("  -> Overfitting detected! Try lower C or different kernel.")

        print(f"\n5-Fold Cross-Validation (on training set):")
        cv_scores = cross_val_score(self.svm_model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        self.cv_scores['SVM'] = cv_scores
        print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
        print(f"  Individual folds: {[f'{s*100:.1f}%' for s in cv_scores]}")

        return y_test_pred

    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path, normalize=False):
        """
        Generate and save a confusion matrix figure.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name for the plot title
            save_path: Where to save the figure
            normalize: If True, each row is divided by its sum (shows recall per class)
        """
        cm = confusion_matrix(y_true, y_pred)
        unique_classes = sorted(np.unique(y_true))
        display_labels = [self.gesture_map[str(int(c))] for c in unique_classes]

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_plot = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)
            values_format = '.2f'
        else:
            cm_plot = cm
            values_format = 'd'

        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_plot, display_labels=display_labels)
        disp.plot(ax=ax, cmap='Blues', values_format=values_format, colorbar=True)

        title = f'Confusion Matrix -- {model_name}'
        if normalize:
            title += ' (Normalized)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  OK Saved to: {save_path}")

        return cm

    def analyze_confusion_matrix(self, cm, model_name):
        """
        Analyze confusion matrix to identify which gestures are most confused.

        Args:
            cm: Confusion matrix (raw counts)
            model_name: Model name for display
        """
        print(f"\n{model_name} - Confusion Analysis:")
        print("-" * 60)

        unique_classes = sorted(np.unique(self.y_test))

        print("Per-class accuracy:")
        for i, class_id in enumerate(unique_classes):
            gesture_name = self.gesture_map[str(int(class_id))]
            class_total = cm[i].sum()
            class_correct = cm[i, i]
            class_accuracy = (class_correct / class_total * 100) if class_total > 0 else 0
            print(f"  {gesture_name:15}: {class_accuracy:5.1f}% ({class_correct}/{class_total})")

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
            for count, true_g, pred_g in confusion_pairs[:5]:
                print(f"  {true_g:15} -> {pred_g:15}: {count} times")
        else:
            print("  No confusion! Perfect classification!")

    def evaluate_all_classifiers(self, predictions):
        """
        Unified evaluation: classification reports, normalized confusion matrices,
        and inference timing for all classifiers.

        Args:
            predictions: dict of {name: (fitted_model, y_pred_array)}

        Returns:
            dict of {name: {accuracy, f1_weighted, train_time_s, inf_time_ms}}
        """
        print("\n" + "="*60)
        print("STEP 8: Classification Reports & Normalized Confusion Matrices")
        print("="*60)

        results = {}

        for name, (model, y_pred) in predictions.items():
            print(f"\n{'-'*40}")
            print(f"  {name}")
            print(f"{'-'*40}")

            # Classification report (per-class precision, recall, F1)
            report_str = classification_report(
                self.y_test, y_pred,
                target_names=self.gesture_names,
                zero_division=0
            )
            print(report_str)

            report_dict = classification_report(
                self.y_test, y_pred,
                target_names=self.gesture_names,
                output_dict=True,
                zero_division=0
            )

            # Normalized confusion matrix
            safe_name = name.replace(' ', '_').lower()
            norm_path = self.models_dir / f"confusion_matrix_{safe_name}_normalized.png"
            self.plot_confusion_matrix(
                self.y_test, y_pred,
                model_name=name,
                save_path=norm_path,
                normalize=True
            )

            # Inference time: average over 100 passes x test set size
            t0 = time.perf_counter()
            for _ in range(100):
                model.predict(self.X_test)
            inf_time_ms = (time.perf_counter() - t0) / (100 * len(self.X_test)) * 1000

            results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'f1_weighted': report_dict['weighted avg']['f1-score'],
                'train_time_s': self.train_times.get(name, 0.0),
                'inf_time_ms': inf_time_ms,
            }

        # Summary timing table
        print(f"\n{'-'*60}")
        print(f"  Timing Summary")
        print(f"{'-'*60}")
        print(f"  {'Classifier':<20} {'Train Time (s)':<18} {'Inference (ms/sample)'}")
        print(f"  {'-'*55}")
        for name, r in results.items():
            print(f"  {name:<20} {r['train_time_s']:<18.3f} {r['inf_time_ms']:.4f}")

        return results

    def plot_comparison_bar(self, results):
        """
        Grouped bar chart comparing accuracy and weighted F1 across all classifiers.

        Args:
            results: dict returned by evaluate_all_classifiers()
        """
        names = list(results.keys())
        accuracies = [results[n]['accuracy'] * 100 for n in names]
        f1s = [results[n]['f1_weighted'] * 100 for n in names]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        bars1 = ax.bar(x - width / 2, accuracies, width, label='Test Accuracy', color='steelblue')
        bars2 = ax.bar(x + width / 2, f1s, width, label='Weighted F1', color='darkorange')

        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Classifier Comparison -- Accuracy and Weighted F1', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylim(0, 115)
        ax.legend(fontsize=10)
        ax.bar_label(bars1, fmt='%.1f%%', padding=3, fontsize=9)
        ax.bar_label(bars2, fmt='%.1f%%', padding=3, fontsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

        plt.tight_layout()
        save_path = self.models_dir / 'classifier_comparison_bar.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  OK Saved to: {save_path}")

    def plot_learning_curves(self, models):
        """
        Plot train vs. validation accuracy as training set size grows.

        Shows the bias-variance tradeoff and whether more data would help.

        Args:
            models: dict of {name: fitted_model_instance}
                    sklearn's learning_curve clones each model internally,
                    so the originals are not modified.
        """
        print(f"\nGenerating learning curves (this may take a moment)...")

        n = len(models)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
        if n == 1:
            axes = [axes]

        train_sizes_pct = np.linspace(0.1, 1.0, 8)

        for ax, (name, model) in zip(axes, models.items()):
            train_sizes, train_scores, val_scores = learning_curve(
                model,
                self.X_train, self.y_train,
                train_sizes=train_sizes_pct,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )

            train_mean = train_scores.mean(axis=1) * 100
            train_std = train_scores.std(axis=1) * 100
            val_mean = val_scores.mean(axis=1) * 100
            val_std = val_scores.std(axis=1) * 100

            ax.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Training', linewidth=1.5)
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                            alpha=0.15, color='steelblue')
            ax.plot(train_sizes, val_mean, 'o-', color='darkorange', label='Validation', linewidth=1.5)
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                            alpha=0.15, color='darkorange')

            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.set_xlabel('Training samples', fontsize=10)
            ax.set_ylim(50, 105)
            ax.yaxis.grid(True, linestyle='--', alpha=0.4)
            ax.set_axisbelow(True)
            ax.legend(loc='lower right', fontsize=9)

        axes[0].set_ylabel('Accuracy (%)', fontsize=10)
        fig.suptitle('Learning Curves', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = self.models_dir / 'learning_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  OK Saved to: {save_path}")

    def statistical_significance_tests(self):
        """
        Paired t-test on 5-fold CV scores for every classifier pair.

        Each classifier produces 5 accuracy values (one per fold). A paired
        t-test tests whether the mean difference across folds is significantly
        different from zero (H0: classifiers perform equally).
        """
        print("\n" + "="*60)
        print("STEP 10: Statistical Significance Tests")
        print("="*60)
        print("\nPaired t-test on 5-fold CV accuracy scores (alpha = 0.05):")
        print(f"  {'Pair':<35} {'t-statistic':>12} {'p-value':>10} {'Result'}")
        print(f"  {'-'*70}")

        names = list(self.cv_scores.keys())
        any_significant = False

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                t_stat, p_val = stats.ttest_rel(self.cv_scores[a], self.cv_scores[b])
                significant = p_val < 0.05
                if significant:
                    any_significant = True
                result = "SIGNIFICANT *" if significant else "not significant"
                pair_label = f"{a} vs {b}"
                print(f"  {pair_label:<35} {t_stat:>12.3f} {p_val:>10.4f}   {result}")

        if not any_significant:
            print("\n  No pairwise difference is statistically significant.")
            print("  This suggests all three classifiers perform comparably on this dataset.")
        else:
            print("\n  * p < 0.05: the difference is unlikely due to chance.")

    def save_models(self):
        """Save trained models to disk."""
        print("\n" + "="*60)
        print("STEP 11: Saving Models")
        print("="*60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.rf_model:
            rf_path = self.models_dir / f"random_forest_{timestamp}.pkl"
            with open(rf_path, 'wb') as f:
                pickle.dump(self.rf_model, f)
            print(f"OK Random Forest saved to: {rf_path}")

        if self.knn_model:
            knn_path = self.models_dir / f"knn_{timestamp}.pkl"
            with open(knn_path, 'wb') as f:
                pickle.dump(self.knn_model, f)
            print(f"OK k-NN saved to: {knn_path}")

        if self.svm_model:
            svm_path = self.models_dir / f"svm_{timestamp}.pkl"
            with open(svm_path, 'wb') as f:
                pickle.dump(self.svm_model, f)
            print(f"OK SVM saved to: {svm_path}")

        # RF is the production model
        best_model_path = self.models_dir / "gesture_classifier_latest.pkl"
        with open(best_model_path, 'wb') as f:
            pickle.dump(self.rf_model, f)
        print(f"OK Latest model (RF) saved to: {best_model_path}")

    def run_full_pipeline(self):
        """Run the complete training and evaluation pipeline."""
        print("\n" + "="*60)
        print("HAND GESTURE RECOGNITION - MODEL TRAINING PIPELINE")
        print("="*60)

        # Steps 1-3: data
        df = self.load_and_merge_data()
        self.prepare_features_and_labels(df)
        self.split_train_test(test_size=0.2, random_state=42)

        # Steps 4-6: train all classifiers
        rf_predictions  = self.train_random_forest(n_estimators=100, max_depth=None, min_samples_split=2)
        knn_predictions = self.train_knn(n_neighbors=5)
        svm_predictions = self.train_svm(C=10, gamma='scale')

        # Step 7: raw confusion matrices + per-class analysis (existing behaviour)
        print("\n" + "="*60)
        print("STEP 7: Raw Confusion Matrices")
        print("="*60)

        for name, preds, safe in [
            ("Random Forest", rf_predictions, "rf"),
            ("KNN",           knn_predictions, "knn"),
            ("SVM",           svm_predictions, "svm"),
        ]:
            print(f"\n{name}:")
            cm = self.plot_confusion_matrix(
                self.y_test, preds,
                model_name=name,
                save_path=self.models_dir / f"confusion_matrix_{safe}.png",
                normalize=False
            )
            self.analyze_confusion_matrix(cm, name)

        # Step 8: classification reports + normalized CMs + timing
        predictions = {
            'Random Forest': (self.rf_model,  rf_predictions),
            'KNN':           (self.knn_model, knn_predictions),
            'SVM':           (self.svm_model, svm_predictions),
        }
        results = self.evaluate_all_classifiers(predictions)

        # Step 9: comparison bar chart + learning curves
        print("\n" + "="*60)
        print("STEP 9: Comparative Visualizations")
        print("="*60)
        self.plot_comparison_bar(results)

        models_for_lc = {
            'Random Forest': self.rf_model,
            'KNN':           self.knn_model,
            'SVM':           self.svm_model,
        }
        self.plot_learning_curves(models_for_lc)

        # Step 10: statistical significance
        self.statistical_significance_tests()

        # Step 11: save
        self.save_models()

        # Final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\n{'Classifier':<20} {'Test Accuracy':>15} {'Weighted F1':>13} {'CV Mean +/- Std':>18}")
        print(f"{'-'*68}")
        for name, r in results.items():
            cv = self.cv_scores[name]
            print(f"{name:<20} {r['accuracy']*100:>14.2f}%  {r['f1_weighted']*100:>12.2f}%  "
                  f"{cv.mean()*100:>7.2f}% +/- {cv.std()*100:.2f}%")

        best = max(results, key=lambda n: results[n]['accuracy'])
        print(f"\nBest classifier by test accuracy: {best} ({results[best]['accuracy']*100:.2f}%)")
        print(f"\nGenerated artifacts in models/:")
        for name in ['rf', 'knn', 'svm']:
            print(f"  confusion_matrix_{name}.png")
            print(f"  confusion_matrix_{name}_normalized.png")
        print(f"  classifier_comparison_bar.png")
        print(f"  learning_curves.png")


def main():
    """Main entry point."""
    trainer = GestureModelTrainer()
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()
