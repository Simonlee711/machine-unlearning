# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve
)
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import random  # Importing random module
import os  # For checking file existence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the DecisionTree class
class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, gradients, hessians):
        self.features = X.columns
        self.tree = self._build_tree(X, y, gradients, hessians, depth=0)

    def _build_tree(self, X, y, gradients, hessians, depth):
        num_samples = X.shape[0]
        if depth >= self.max_depth or num_samples < self.min_samples_split:
            grad_sum = np.sum(gradients)
            hess_sum = np.sum(hessians)
            leaf_weight = -grad_sum / (hess_sum + 1e-6)
            return {'type': 'leaf', 'weight': leaf_weight, 'samples': X.index.tolist()}

        best_feature, best_threshold, best_gain = self._find_best_split(X, gradients, hessians)
        if best_gain is None or best_gain <= 0:
            grad_sum = np.sum(gradients)
            hess_sum = np.sum(hessians)
            leaf_weight = -grad_sum / (hess_sum + 1e-6)
            return {'type': 'leaf', 'weight': leaf_weight, 'samples': X.index.tolist()}

        left_indices = X[best_feature] <= best_threshold
        right_indices = X[best_feature] > best_threshold

        left_subtree = self._build_tree(
            X[left_indices], y[left_indices], gradients[left_indices], hessians[left_indices], depth + 1
        )
        right_subtree = self._build_tree(
            X[right_indices], y[right_indices], gradients[right_indices], hessians[right_indices], depth + 1
        )

        return {
            'type': 'node',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def _find_best_split(self, X, gradients, hessians):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in self.features:
            unique_values = np.unique(X[feature])
            if len(unique_values) <= 1:
                continue
            thresholds = unique_values[:-1]  # Exclude the last value to prevent empty splits
            for threshold in thresholds:
                left = X[feature] <= threshold
                right = X[feature] > threshold

                if np.sum(left) == 0 or np.sum(right) == 0:
                    continue

                left_grad = np.sum(gradients[left])
                right_grad = np.sum(gradients[right])
                left_hess = np.sum(hessians[left])
                right_hess = np.sum(hessians[right])

                gain = (
                    self._calculate_gain(left_grad, left_hess) +
                    self._calculate_gain(right_grad, right_hess) -
                    self._calculate_gain(np.sum(gradients), np.sum(hessians))
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_gain > 0:
            return best_feature, best_threshold, best_gain
        else:
            return None, None, None

    def _calculate_gain(self, grad_sum, hess_sum):
        return (grad_sum ** 2) / (hess_sum + 1e-6)

    def predict_single(self, x, node=None):
        if node is None:
            node = self.tree
        if node['type'] == 'leaf':
            return node['weight']
        if x[node['feature']] <= node['threshold']:
            return self.predict_single(x, node['left'])
        else:
            return self.predict_single(x, node['right'])

    def update_leaf(self, samples_to_remove, g_remove, h_remove):
        """
        Update the leaf node weights by removing the contributions of certain samples.
        This is a simplified implementation and may need to be adapted based on your tree structure.
        """
        def traverse(node):
            if node['type'] == 'leaf':
                # Check which samples in this leaf are being removed
                removed_in_leaf = [idx for idx in samples_to_remove if idx in node['samples']]
                if removed_in_leaf:
                    # Calculate the new gradient and hessian sums after removal
                    current_grad_sum = np.sum(g_remove[node['samples']])
                    current_hess_sum = np.sum(h_remove[node['samples']])
                    removed_grad_sum = np.sum(g_remove[removed_in_leaf])
                    removed_hess_sum = np.sum(h_remove[removed_in_leaf])

                    new_grad_sum = current_grad_sum - removed_grad_sum
                    new_hess_sum = current_hess_sum - removed_hess_sum

                    # Update the leaf weight
                    node['weight'] = -new_grad_sum / (new_hess_sum + 1e-6)
            else:
                traverse(node['left'])
                traverse(node['right'])

        traverse(self.tree)

# Define the GradientBoostedTrees class
class GradientBoostedTrees:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, min_samples_split=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.init_score = 0.0
        self.sample_gradients = []
        self.sample_hessians = []
        self.F = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        pos = np.sum(y)
        neg = len(y) - pos
        self.init_score = np.log((pos + 1) / (neg + 1e-6))
        self.F = np.full(len(y), self.init_score)

        for i in tqdm(range(self.n_estimators), desc="Training Trees"):
            p = self.sigmoid(self.F)
            g = p - y
            h = p * (1 - p)

            # Store gradients and hessians as NumPy arrays
            self.sample_gradients.append(g.copy())
            self.sample_hessians.append(h.copy())

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, y, g, h)
            self.trees.append(tree)

            # Update F using vectorized operations for efficiency
            predictions = X.apply(lambda row: tree.predict_single(row), axis=1).values
            self.F += self.learning_rate * predictions

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.init_score)
        for tree in self.trees:
            predictions = X.apply(lambda row: tree.predict_single(row), axis=1).values
            F += self.learning_rate * predictions
        return self.sigmoid(F)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def unlearn(self, X, y, indices_to_remove):
        logging.info(f"Starting unlearning of {len(indices_to_remove)} samples.")

        for tree_idx, tree in enumerate(self.trees):
            g = self.sample_gradients[tree_idx]
            h = self.sample_hessians[tree_idx]
            # Ensure indices_to_remove are within the current gradients array
            samples_in_tree = [idx for idx in indices_to_remove if idx < len(g)]

            if not samples_in_tree:
                continue

            # Update the leaf nodes by removing the contributions of the samples to remove
            tree.update_leaf(samples_in_tree, g, h)

            # Zero out the gradients and hessians for the removed samples
            g[samples_in_tree] = 0.0
            h[samples_in_tree] = 0.0

        logging.info("Recomputing predictions after unlearning.")
        self.F = np.full(len(y), self.init_score)
        for tree in self.trees:
            predictions = X.apply(lambda row: tree.predict_single(row), axis=1).values
            self.F += self.learning_rate * predictions

        logging.info("Unlearning process completed.")

# Load a small sample of the dataset
file_path = './X_mimic_iv.csv.gz'  # Update with your file path

# For demonstration purposes, let's create a mock dataset
# Remove the following block if you have the actual dataset
if not os.path.exists(file_path):
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
        'death': np.random.randint(0, 2, 1000),
        'subject_id': np.arange(1000),
        'time': np.random.randint(1, 100, 1000),
        'Unnamed: 0.1': np.random.randn(1000),
        'gender': np.random.choice(['M', 'F'], 1000)
    })
    data.to_csv(file_path, index=False, compression='gzip')

# Read the dataset
data = pd.read_csv(file_path, compression='gzip')
data = data.sample(100)
# Preprocess the dataset
X = data.drop(columns=['death', 'subject_id', 'time', 'Unnamed: 0.1', 'gender'])
y = data['death']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Reset indices to ensure they start from 0
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Apply scaling
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X.columns,
    index=X_test.index
)

# Train and evaluate the model
gbt_model = GradientBoostedTrees(
    n_estimators=5,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=10
)
gbt_model.fit(X_train_scaled, y_train)

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    preds = model.predict_proba(X_test)
    preds_binary = (preds >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds_binary)
    prec = precision_score(y_test, preds_binary, zero_division=0)
    rec = recall_score(y_test, preds_binary, zero_division=0)
    f1 = f1_score(y_test, preds_binary, zero_division=0)
    ll = log_loss(y_test, preds)
    return acc, prec, rec, f1, ll

test_results = evaluate_model(gbt_model, X_test_scaled, y_test)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Log Loss']
test_performance = dict(zip(metrics, test_results))
print("Test Performance:", test_performance)

# Function to simulate membership inference attack and calculate AUC
def membership_inference_attack_auc(model, X, labels):
    """
    Performs a membership inference attack by comparing the predictions of removed
    and non-removed samples. Returns the AUC and ROC curve data.
    """
    predictions = model.predict_proba(X)

    # Generate the attack predictions (confidence of membership)
    attack_confidences = predictions

    # Compute AUC
    auc = roc_auc_score(labels, attack_confidences)
    fpr, tpr, thresholds = roc_curve(labels, attack_confidences)

    return auc, fpr, tpr

# Unlearning a percentage of data
unlearning_percentage = 0.1  # 10% of the training data
num_samples_to_remove = max(1, int(unlearning_percentage * len(X_train_scaled)))  # Ensure at least one sample is removed
removed_indices = random.sample(list(X_train_scaled.index), num_samples_to_remove)

# Store the removed samples before dropping
X_removed = X_train_scaled.loc[removed_indices].copy()
y_removed = y_train.loc[removed_indices].copy()

# Perform unlearning on the original model
gbt_model.unlearn(X_train_scaled, y_train, removed_indices)

# Train a baseline model on the remaining data
X_retrained = X_train_scaled.drop(index=removed_indices).reset_index(drop=True)
y_retrained = y_train.drop(index=removed_indices).reset_index(drop=True)

baseline_model = GradientBoostedTrees(
    n_estimators=5,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=10
)
baseline_model.fit(X_retrained, y_retrained)

# Prepare attack datasets

# For Original Model:
# Assume attacker has access to all training data and wants to know if a sample was removed
# Labels: 1 for removed samples, 0 for non-removed samples
attack_X_original = X_train_scaled.copy()
labels_original = np.zeros(len(attack_X_original))
labels_original[removed_indices] = 1

# For Baseline Model:
# The baseline model was trained without the removed samples
# To perform a fair attack, combine the remaining training samples with the removed samples
attack_X_baseline = pd.concat([X_retrained, X_removed], ignore_index=True)
labels_baseline = np.concatenate([np.zeros(len(X_retrained)), np.ones(len(X_removed))])

# Perform membership inference attack on Original Model
auc_original, fpr_original, tpr_original = membership_inference_attack_auc(
    gbt_model, attack_X_original, labels_original
)

# Perform membership inference attack on Baseline Model
auc_baseline, fpr_baseline, tpr_baseline = membership_inference_attack_auc(
    baseline_model, attack_X_baseline, labels_baseline
)

# Plot the ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_original, tpr_original, label=f"Original Model ROC (AUC = {auc_original:.2f})")
plt.plot(fpr_baseline, tpr_baseline, label=f"Baseline Model ROC (AUC = {auc_baseline:.2f})", linestyle='--')
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Membership Inference Attack ROC Curves")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# Display AUC scores
print(f"Membership Inference Attack AUC (Original Model): {auc_original:.2f}")
print(f"Membership Inference Attack AUC (Baseline Model): {auc_baseline:.2f}")

# %%
