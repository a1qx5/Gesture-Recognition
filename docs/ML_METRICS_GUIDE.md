# ML Evaluation Metrics & Graphs Guide

This document explains every statistical metric and graph used in `src/train_model.py` and `src/analyze_model.py` — what each one measures, how it is calculated, and how to interpret it in general terms.

---

## Table of Contents

1. [The Train/Test Split](#1-the-traintest-split)
2. [Accuracy](#2-accuracy)
3. [The Train/Test Gap & Overfitting](#3-the-traintest-gap--overfitting)
4. [Confusion Matrix](#4-confusion-matrix)
5. [Precision](#5-precision)
6. [Recall (Sensitivity)](#6-recall-sensitivity)
7. [F1-Score](#7-f1-score)
8. [Weighted F1-Score](#8-weighted-f1-score)
9. [Support](#9-support)
10. [K-Fold Cross-Validation](#10-k-fold-cross-validation)
11. [Learning Curves](#11-learning-curves)
12. [Paired T-Test (Statistical Significance)](#12-paired-t-test-statistical-significance)
13. [Inference Time](#13-inference-time)
14. [Feature Importance](#14-feature-importance)
15. [Data Distribution Chart](#15-data-distribution-chart)
16. [Classifier Comparison Bar Chart](#16-classifier-comparison-bar-chart)

---

## 1. The Train/Test Split

Before training, the dataset is split into two non-overlapping subsets:

- **Training set** — the data the model learns from
- **Test set** — data the model has never seen, used only to evaluate it after training

**Why split at all?** A model evaluated on the same data it trained on will always appear accurate — it has simply memorized the answers. The test set simulates real-world usage by presenting the model with genuinely unseen examples.

**Stratification** means the class distribution (proportion of each gesture) is preserved in both splits. Without stratification, a random split might accidentally put most examples of one class into the training set and few into the test set, making evaluation misleading.

**Interpretation:**
- A model that performs well on training data but poorly on test data has a problem (see section 3)
- A model that performs similarly on both sets is generalizing correctly

---

## 2. Accuracy

**What it is:** The proportion of all predictions that were correct.

**Formula:**
```
Accuracy = Correct Predictions / Total Predictions
```

Or equivalently:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Where TP = true positives, TN = true negatives, FP = false positives, FN = false negatives.

**Interpretation:**
- Range: 0.0 to 1.0 (or 0% to 100%)
- Higher is better
- 1.0 = perfect, every prediction was correct
- 0.0 = every prediction was wrong

**Limitation:** Accuracy can be misleading when classes are imbalanced. If 90% of your data is class A and your model always predicts class A, it achieves 90% accuracy while being completely useless for all other classes. F1-score (section 7) is more informative in that case.

---

## 3. The Train/Test Gap & Overfitting

The gap is simply:
```
Gap = Train Accuracy - Test Accuracy
```

This measures how much worse the model performs on unseen data compared to data it trained on.

**Interpretation:**
- **Gap ≈ 0** — the model generalizes well. What it learned transfers to new data.
- **Small gap (e.g. < 5%)** — acceptable. Some gap is normal and expected.
- **Large gap (e.g. > 10%)** — **overfitting**. The model has memorized the training data rather than learning generalizable patterns. It performs well on training data but fails on new data.
- **Negative gap (test > train)** — rare, usually means the test set happened to be easier than the training set, or the training set is very small.

**Overfitting** is the most common failure mode in machine learning. Causes include: too complex a model, too little training data, or not enough regularization (constraints on the model's complexity).

---

## 4. Confusion Matrix

A square grid where:
- **Rows** represent the **actual** (true) class
- **Columns** represent the **predicted** class
- Each cell `[i][j]` contains the count of times class `i` was predicted as class `j`

**Example (3 classes):**
```
              Predicted A  Predicted B  Predicted C
Actual A   [     45           2            3      ]
Actual B   [      1          48            1      ]
Actual C   [      0           3           47      ]
```

**Interpretation:**
- The **diagonal** (top-left to bottom-right) = correct predictions. You want these to be high.
- **Off-diagonal cells** = errors. Cell `[i][j]` where `i ≠ j` means the model confused class `i` for class `j`.
- A large off-diagonal value between two classes means those two classes are frequently mistaken for each other — they share similar features from the model's perspective.

**Normalized confusion matrix:** Each row is divided by the total number of actual instances in that class, converting counts to percentages (0.0–1.0). This makes it easier to compare classes with different sample counts — a row with 10 samples and a row with 100 samples become directly comparable.

---

## 5. Precision

**What it is:** Of all the times the model predicted class X, what fraction actually were class X?

**Formula:**
```
Precision = TP / (TP + FP)
```

- TP (True Positive) = model predicted X, it was actually X ✓
- FP (False Positive) = model predicted X, it was actually something else ✗

**Interpretation:**
- Range: 0.0 to 1.0
- High precision = when the model says "this is class X," it is usually right
- Low precision = the model predicts X too often, including for things that aren't X
- Precision = 1.0 means every time the model predicted X, it was correct (but it may have missed many actual X instances)

**Analogy:** A spam filter with high precision rarely marks real emails as spam (few false positives), but might let some spam through.

---

## 6. Recall (Sensitivity)

**What it is:** Of all the actual instances of class X in the data, what fraction did the model correctly find?

**Formula:**
```
Recall = TP / (TP + FN)
```

- TP (True Positive) = model predicted X, it was actually X ✓
- FN (False Negative) = it was actually X, but the model missed it ✗

**Interpretation:**
- Range: 0.0 to 1.0
- High recall = the model finds most actual instances of X
- Low recall = the model misses many actual instances of X
- Recall = 1.0 means the model never missed a real X (but it may have falsely predicted X for many non-X instances)

**Analogy:** A spam filter with high recall catches almost all spam (few false negatives), but might also flag some real emails.

---

## 7. F1-Score

**What it is:** The harmonic mean of precision and recall, combining both into a single number.

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Why harmonic mean and not a regular average?** The harmonic mean punishes extreme imbalances between precision and recall. A model with precision=1.0 and recall=0.0 would have an arithmetic mean of 0.5 (misleadingly decent) but an F1 of 0.0 (correctly terrible).

**Interpretation:**
- Range: 0.0 to 1.0
- Higher is better
- F1 = 1.0 means both precision and recall are perfect
- F1 = 0.0 means either precision or recall is 0
- A good F1 score requires *both* precision and recall to be good — you cannot game it by sacrificing one for the other
- More informative than accuracy alone when classes are imbalanced

**Precision vs Recall tradeoff:** There is often a fundamental tension between the two — tuning a model to be more precise typically reduces recall and vice versa. F1 sits at the balance point between them.

---

## 8. Weighted F1-Score

**What it is:** The F1-score averaged across all classes, where each class's F1 is weighted by how many actual instances of that class exist in the test set (its "support").

**Formula:**
```
Weighted F1 = Σ (F1_class_i × support_i) / total_samples
```

**Interpretation:**
- Same range (0.0–1.0) and direction (higher = better) as regular F1
- Classes with more samples contribute more to the final score
- More representative of real-world performance than a simple average across classes, since it accounts for class frequency
- If all classes have equal support, weighted F1 equals the simple (macro) average F1

---

## 9. Support

**What it is:** The number of actual instances of a class in the test set. Not a performance metric — just a count.

**Why it matters:** Gives context to precision/recall/F1 values. A class with support=2 and F1=0.5 is far less statistically meaningful than a class with support=100 and F1=0.5 — the former could easily be a fluke of two unlucky test samples.

---

## 10. K-Fold Cross-Validation

**What it is:** A more robust evaluation strategy than a single train/test split. The training data is divided into `k` equal "folds." The model is trained and evaluated `k` times — each time using one fold as the validation set and the remaining `k-1` folds as training data. The results across all `k` runs are then averaged.

**With k=5 (as used in this project):**
```
Run 1: Train on folds 2,3,4,5 → Validate on fold 1
Run 2: Train on folds 1,3,4,5 → Validate on fold 2
Run 3: Train on folds 1,2,4,5 → Validate on fold 3
Run 4: Train on folds 1,2,3,5 → Validate on fold 4
Run 5: Train on folds 1,2,3,4 → Validate on fold 5
→ Report: mean ± standard deviation of the 5 scores
```

**Why it is better than a single split:**
- A single split result depends heavily on which samples happened to land in the test set — a lucky or unlucky split can make a model look better or worse than it really is
- Cross-validation uses every sample as a test sample exactly once, giving a more stable and representative estimate of true performance

**Interpretation:**
- **Mean** — the best estimate of how well the model generalizes. Higher is better.
- **Standard deviation (std)** — how consistent the model is across different data subsets
  - Low std (e.g. ±0.01) = stable, reliable performance regardless of which data it sees
  - High std (e.g. ±0.10) = performance varies significantly depending on the data split — possibly overfitting to some subsets, or the dataset has meaningful variation between folds

---

## 11. Learning Curves

**What they are:** A plot of model accuracy (y-axis) against training set size (x-axis), showing two lines:
- **Training score** — accuracy on the training data itself
- **Validation score** — accuracy on held-out data (cross-validated)

Both lines are shown with a shaded band representing ±1 standard deviation across CV folds.

**How to read them:**

**Converging lines (good generalization):**
```
Accuracy
  |  train ─────────────────────────
  |                      ╲──────────  validation
  |
  └─────────────────────────────────▶ Training size
```
Training and validation scores converge as more data is added. The model generalizes well. Adding more data would provide diminishing returns.

**Diverging lines (overfitting):**
```
Accuracy
  |  train ─────────────────────────
  |
  |
  |  validation ─────────────────────
  └─────────────────────────────────▶ Training size
```
A persistent large gap that does not close as training size grows. The model memorizes training data but fails to generalize. Solutions: simplify the model, add regularization, or collect more diverse data.

**Both lines low (underfitting):**
```
Accuracy
  |
  |
  |  train ──────────────────────────
  |  validation ─────────────────────
  └─────────────────────────────────▶ Training size
```
The model is too simple to capture the patterns in the data. Both scores are low regardless of how much data is available. Solution: use a more complex model.

**Validation still rising at the right edge:**
The validation score has not plateaued — adding more training data would likely improve performance further.

---

## 12. Paired T-Test (Statistical Significance)

**What it is:** A statistical test that answers: "Is the performance difference between two classifiers real, or could it be due to random chance in how the data was split?"

**How it works here:** Each classifier produces `k` scores from k-fold cross-validation (5 scores in this project). The paired t-test compares those two arrays of scores directly — "paired" because each score came from the same fold, so the comparison is fair (same data subset, different model).

**The null hypothesis (H₀):** The two classifiers have the same true performance — any observed difference is just random noise.

**The two outputs:**

**t-statistic:**
- Measures the size and direction of the difference relative to the variability across folds
- `t > 0` — the first classifier scored higher on average
- `t < 0` — the second classifier scored higher on average
- Larger absolute value = larger difference relative to the noise/variance in scores

**p-value:**
- The probability of observing a difference this large (or larger) purely by chance, *assuming* the null hypothesis is true
- `p < 0.05` — statistically significant at the 5% level: reject the null hypothesis, conclude the classifiers genuinely differ in performance
- `p ≥ 0.05` — fail to reject the null hypothesis: the observed difference could plausibly be due to chance alone
- `p < 0.01` — stronger evidence of a real difference
- `p < 0.001` — very strong evidence

**α (alpha) = 0.05** is the conventional significance threshold — accepting a 5% chance of falsely concluding a difference exists when it doesn't (a Type I error).

**Important distinction:** Statistical significance ≠ practical significance. A p-value of 0.001 confirms the difference is real, not that it matters. A difference of 0.1% accuracy may be statistically significant with enough data but practically irrelevant.

---

## 13. Inference Time

**What it is:** How long the model takes to make a single prediction, measured in milliseconds.

**How it is measured here:** The model predicts on the entire test set 100 times in a loop; the total wall-clock time is divided by `(100 × number of test samples)` to get a per-sample average.

**Interpretation:**
- Lower is better for real-time applications
- The absolute scale matters: 0.1ms/prediction is excellent for real-time use; 100ms/prediction would cap throughput at ~10 predictions per second
- Unlike accuracy metrics, inference time is hardware-dependent — the same model runs faster on better hardware

---

## 14. Feature Importance

**What it is:** A property specific to Random Forest (and tree-based models generally). Each feature is assigned an importance score reflecting how much it contributed to the model's decisions across all trees.

**How it is calculated:** When a tree splits on a feature, it measures how much that split reduced impurity (uncertainty about the class) in the training samples. This reduction is averaged over all splits using that feature, across all trees in the forest. The resulting importances are normalized to sum to 1.0.

**Interpretation:**
- Range: 0.0 to 1.0 per feature; all features sum to 1.0
- Higher = that feature contributed more to the model's decisions
- Near 0 = the model almost never found that feature useful for splitting
- Features with high importance are the most discriminative signals — the ones that most reliably differ between classes

**What it tells you about your data:** High importance on a particular feature means that variable is highly discriminative. Low importance means it barely differs across classes and the model essentially ignores it.

**Limitation:** When two features are highly correlated, their importance gets split between them, making each look less important than it actually is collectively.

---

## 15. Data Distribution Chart

**What it is:** A bar chart and pie chart showing how many samples exist per class in the dataset.

**Interpretation:**
- **Balanced distribution** — roughly equal samples per class. Ideal: accuracy, F1, and confusion matrices are all straightforward to interpret.
- **Imbalanced distribution** — some classes have many more samples than others. This can bias the model toward predicting majority classes more often, inflating accuracy while recall on minority classes suffers. Weighted F1 (section 8) is more meaningful than plain accuracy in this case.
- The pie chart shows proportional share; the bar chart shows absolute counts.

---

## 16. Classifier Comparison Bar Chart

**What it is:** A grouped bar chart placing multiple classifiers side by side, with two bars per classifier — one for test accuracy and one for weighted F1.

**How to read it:**
- Taller bars = better performance
- The gap between a classifier's accuracy bar and its F1 bar indicates how much class imbalance is distorting the accuracy metric — if accuracy >> F1, the model is leaning on majority classes and underperforming on minority ones
- Comparing across classifiers: a classifier whose two bars are both tall and close together is performing well and consistently across all classes
- Percentage labels on each bar make exact values easy to read without eyeballing the y-axis
