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
import random
import os
import time

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
        def traverse(node):
            if node['type'] == 'leaf':
                removed_in_leaf = [idx for idx in samples_to_remove if idx in node['samples']]
                if removed_in_leaf:
                    current_grad_sum = np.sum(g_remove[node['samples']])
                    current_hess_sum = np.sum(h_remove[node['samples']])
                    removed_grad_sum = np.sum(g_remove[removed_in_leaf])
                    removed_hess_sum = np.sum(h_remove[removed_in_leaf])

                    new_grad_sum = current_grad_sum - removed_grad_sum
                    new_hess_sum = current_hess_sum - removed_hess_sum

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

            self.sample_gradients.append(g.copy())
            self.sample_hessians.append(h.copy())

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, y, g, h)
            self.trees.append(tree)

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
        for tree_idx, tree in enumerate(self.trees):
            g = self.sample_gradients[tree_idx]
            h = self.sample_hessians[tree_idx]
            samples_in_tree = [idx for idx in indices_to_remove if idx < len(g)]

            if not samples_in_tree:
                continue

            tree.update_leaf(samples_in_tree, g, h)

            g[samples_in_tree] = 0.0
            h[samples_in_tree] = 0.0

        self.F = np.full(len(y), self.init_score)
        for tree in self.trees:
            predictions = X.apply(lambda row: tree.predict_single(row), axis=1).values
            self.F += self.learning_rate * predictions

# Generate or load the dataset
file_path = './X_mimic_iv.csv.gz'
data = pd.read_csv(file_path, compression='gzip')
data = data.sample(1000)

# Preprocess the dataset
X = data.drop(columns=['death', 'subject_id', 'time', 'Unnamed: 0.1', 'gender'])
y = data['death']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

# Train the initial model
gbt_model = GradientBoostedTrees(n_estimators=5, learning_rate=0.1, max_depth=3, min_samples_split=10)
gbt_model.fit(X_train_scaled, y_train)

# Runtime analysis experiment
def runtime_experiment(X_train_scaled, y_train, percentages):
    unlearning_times = []
    retraining_times = []

    for perc in percentages:
        num_samples_to_remove = max(1, int(perc * len(X_train_scaled)))
        removed_indices = random.sample(list(X_train_scaled.index), num_samples_to_remove)

        # Unlearning
        gbt_model_copy = GradientBoostedTrees(n_estimators=5, learning_rate=0.1, max_depth=3, min_samples_split=10)
        gbt_model_copy.fit(X_train_scaled, y_train)

        start_unlearning = time.time()
        gbt_model_copy.unlearn(X_train_scaled, y_train, removed_indices)
        unlearning_times.append(time.time() - start_unlearning)

        # Retraining
        X_retrained = X_train_scaled.drop(index=removed_indices).reset_index(drop=True)
        y_retrained = y_train.drop(index=removed_indices).reset_index(drop=True)

        start_retraining = time.time()
        retrained_model = GradientBoostedTrees(n_estimators=5, learning_rate=0.1, max_depth=3, min_samples_split=10)
        retrained_model.fit(X_retrained, y_retrained)
        retraining_times.append(time.time() - start_retraining)

    return unlearning_times, retraining_times

percentages_to_remove = [0.1, 0.2, 0.3, 0.4, 0.5]
unlearning_runtimes, retraining_runtimes = runtime_experiment(X_train_scaled, y_train, percentages_to_remove)

# Plot runtime results
plt.figure(figsize=(10, 6))
plt.plot([p * 100 for p in percentages_to_remove], unlearning_runtimes, label="Unlearning Runtime", marker="o")
plt.plot([p * 100 for p in percentages_to_remove], retraining_runtimes, label="Retraining Runtime", marker="o")
plt.xlabel("Percentage of Data Removed (%)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Analysis: Unlearning vs. Retraining")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# %%
