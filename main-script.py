import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from tqdm import tqdm
import logging
import random
import argparse
import csv
import os

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_binary_classification_datasets_under_1000():
    logging.info("Fetching all active binary classification datasets from OpenML with <1000 instances...")
    # Fetch datasets with binary classification (number_classes=2) and fewer than 1000 instances
    all_ds = openml.datasets.list_datasets(
        output_format='dataframe',
        number_classes=2  # Binary classification
    )
    df = all_ds
    logging.info(f"Available columns in the dataset list: {df.columns.tolist()}")

    # Filter datasets based on the number of instances
    if 'NumberOfInstances' in df.columns:
        df = df[df['NumberOfInstances'] < 1000]
    else:
        logging.warning("Warning: 'NumberOfInstances' column not found. Skipping this filter.")

    # Ensure the target attribute is present
    target_col = 'default_target_attribute' if 'default_target_attribute' in df.columns else None
    if target_col and target_col in df.columns:
        df = df.dropna(subset=[target_col])
    else:
        logging.warning("Warning: 'default_target_attribute' column not found or missing. Skipping this filter.")

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing datasets"):
        did = row['did']
        name = row['name']
        results.append((did, name))
    logging.info(f"Total binary classification datasets after filtering: {len(results)}")
    return results

def load_and_prepare_dataset(did):
    try:
        ds = openml.datasets.get_dataset(did)
        name = ds.name
        target_name = ds.default_target_attribute
        if target_name is None:
            logging.warning(f"Dataset {did} has no default target attribute. Skipping.")
            return None
        X, y, _, _ = ds.get_data(
            target=target_name,
            dataset_format='dataframe'
        )
        n_unique = y.nunique()
        if n_unique != 2:
            logging.info(f"Dataset {did} is not binary. It has {n_unique} classes. Skipping.")
            return None

        # Encode target if necessary
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
            y = pd.Series(y)
        else:
            unique_vals = sorted(y.unique())
            if set(unique_vals) != {0, 1}:
                le = LabelEncoder()
                y = le.fit_transform(y)
                y = pd.Series(y)
            else:
                if not isinstance(y, pd.Series):
                    y = pd.Series(y)

        # Convert object and category dtypes to numerical codes
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = X[col].astype('category').cat.codes

        # Reset index to ensure consistent indexing
        X.reset_index(drop=True, inplace=True)
        y = y.reset_index(drop=True)

        return X, y, name
    except Exception as e:
        logging.error(f"Error loading dataset {did}: {e}")
        return None

class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None  # Will hold the tree structure

    def fit(self, X, y, gradients, hessians):
        self.features = X.columns
        self.tree = self._build_tree(X, y, gradients, hessians, depth=0)

    def _build_tree(self, X, y, gradients, hessians, depth):
        num_samples = X.shape[0]
        if depth >= self.max_depth or num_samples < self.min_samples_split:
            # Leaf node
            grad_sum = np.sum(gradients)
            hess_sum = np.sum(hessians)
            leaf_weight = -grad_sum / (hess_sum + 1e-6)  # Avoid division by zero
            return {
                'type': 'leaf',
                'weight': leaf_weight,
                'grad_sum': grad_sum,
                'hess_sum': hess_sum,
                'samples': X.index.tolist()
            }

        # Find the best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, gradients, hessians)
        if best_gain is None or best_gain <= 0:
            # Cannot find a better split
            grad_sum = np.sum(gradients)
            hess_sum = np.sum(hessians)
            leaf_weight = -grad_sum / (hess_sum + 1e-6)
            return {
                'type': 'leaf',
                'weight': leaf_weight,
                'grad_sum': grad_sum,
                'hess_sum': hess_sum,
                'samples': X.index.tolist()
            }

        # Split the dataset
        left_indices = X[best_feature] <= best_threshold
        right_indices = X[best_feature] > best_threshold

        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_indices], y[left_indices], gradients[left_indices], hessians[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], gradients[right_indices], hessians[right_indices], depth + 1)

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
            unique_values = X[feature].unique()
            if len(unique_values) <= 1:
                continue
            thresholds = unique_values[:-1]  # Avoid the last value to prevent empty split
            for threshold in thresholds:
                left = X[feature] <= threshold
                right = X[feature] > threshold

                if np.sum(left) == 0 or np.sum(right) == 0:
                    continue

                left_grad = np.sum(gradients[left])
                right_grad = np.sum(gradients[right])
                left_hess = np.sum(hessians[left])
                right_hess = np.sum(hessians[right])

                gain = self._calculate_gain(left_grad, left_hess) + self._calculate_gain(right_grad, right_hess) - self._calculate_gain(np.sum(gradients), np.sum(hessians))

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_gain > 0:
            return best_feature, best_threshold, best_gain
        else:
            return None, None, None

    def _calculate_gain(self, grad_sum, hess_sum):
        return (grad_sum ** 2) / (hess_sum + 1e-6)  # Regularization can be added here

    def predict_single(self, x, node=None):
        if node is None:
            node = self.tree
        if node['type'] == 'leaf':
            return node['weight']
        if x[node['feature']] <= node['threshold']:
            return self.predict_single(x, node['left'])
        else:
            return self.predict_single(x, node['right'])

    def get_leaf_mappings(self, X):
        """
        Returns a mapping from sample index to leaf node in this tree.
        """
        mapping = {}
        for idx, row in X.iterrows():
            weight, samples = self._get_leaf_info(row, self.tree)
            mapping[idx] = {'weight': weight, 'samples': samples}
        return mapping

    def _get_leaf_info(self, x, node):
        if node['type'] == 'leaf':
            return node['weight'], node['samples']
        if x[node['feature']] <= node['threshold']:
            return self._get_leaf_info(x, node['left'])
        else:
            return self._get_leaf_info(x, node['right'])

    def update_leaf(self, samples_to_remove, gradients_to_remove, hessians_to_remove):
        """
        Update the leaf weights by removing the contributions of certain samples.
        """
        # Find the leaf containing the samples
        leaf = self._find_leaf(self.tree, samples_to_remove)
        if leaf is not None:
            # Update gradients and hessians
            leaf['grad_sum'] -= gradients_to_remove
            leaf['hess_sum'] -= hessians_to_remove

            # Recompute the leaf weight
            if leaf['hess_sum'] > 0:
                leaf['weight'] = -leaf['grad_sum'] / (leaf['hess_sum'] + 1e-6)
            else:
                leaf['weight'] = 0.0  # Default weight if hess_sum is zero

    def _find_leaf(self, node, samples_to_remove):
        """
        Recursively find the leaf node containing any of the samples to remove.
        """
        if node['type'] == 'leaf':
            if any(sample in node['samples'] for sample in samples_to_remove):
                return node
            else:
                return None
        # Check left subtree
        left_samples = [s for s in samples_to_remove if s in node['left']['samples']]
        if left_samples:
            leaf = self._find_leaf(node['left'], left_samples)
            if leaf:
                return leaf
        # Check right subtree
        right_samples = [s for s in samples_to_remove if s in node['right']['samples']]
        if right_samples:
            leaf = self._find_leaf(node['right'], right_samples)
            if leaf:
                return leaf
        return None

class GradientBoostedTrees:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, min_samples_split=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.init_score = 0.0
        # To store gradients and hessians for each sample and tree
        self.sample_gradients = []  # List of numpy arrays
        self.sample_hessians = []   # List of numpy arrays
        self.F = None               # Raw predictions before sigmoid

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        # Initialize with log(odds)
        pos = np.sum(y)
        neg = len(y) - pos
        self.init_score = np.log((pos + 1) / (neg + 1e-6))
        self.F = np.full(len(y), self.init_score)

        for i in tqdm(range(self.n_estimators), desc="Training Trees"):
            # Compute gradients and hessians
            p = self.sigmoid(self.F)
            g = p - y  # Gradient
            h = p * (1 - p)  # Hessian

            # Store gradients and hessians
            self.sample_gradients.append(g.copy())
            self.sample_hessians.append(h.copy())

            # Fit a decision tree to the gradients
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, y, g, h)
            self.trees.append(tree)

            # Update F
            for idx, row in X.iterrows():
                self.F[idx] += self.learning_rate * tree.predict_single(row)

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.init_score)
        for tree in self.trees:
            for idx, row in X.iterrows():
                F[idx] += self.learning_rate * tree.predict_single(row)
        proba = self.sigmoid(F)
        return proba

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_leaf_mappings(self, X):
        """
        Get leaf mappings for all trees.
        Returns a list of dictionaries for each tree.
        Each dictionary maps sample index to leaf node information.
        """
        mappings = []
        for tree in self.trees:
            mapping = tree.get_leaf_mappings(X)
            mappings.append(mapping)
        return mappings

    def unlearn(self, X, y, indices_to_remove):
        """
        Perform unlearning by removing specific samples.
        """
        logging.info(f"Starting unlearning of {len(indices_to_remove)} samples.")

        for tree_idx, tree in enumerate(self.trees):
            # Get the gradients and hessians for this tree
            g = self.sample_gradients[tree_idx]
            h = self.sample_hessians[tree_idx]

            # Identify the samples to remove in this tree
            # Ensure indices are within bounds
            samples_in_tree = [idx for idx in indices_to_remove if idx < len(g)]
            # Further filter to ensure the sample is in a leaf
            samples_in_tree = [idx for idx in samples_in_tree if tree.tree and tree.tree.get('type') == 'leaf' and idx in tree.tree.get('samples', [])]
            if not samples_in_tree:
                continue  # No samples to remove in this tree

            # Aggregate gradients and hessians to remove
            g_remove = np.sum(g[samples_in_tree])
            h_remove = np.sum(h[samples_in_tree])

            # Update the leaf containing these samples
            tree.update_leaf(samples_in_tree, g_remove, h_remove)

            # Zero out the gradients and hessians for the removed samples
            self.sample_gradients[tree_idx][samples_in_tree] = 0.0
            self.sample_hessians[tree_idx][samples_in_tree] = 0.0

        # Recompute predictions F after unlearning
        logging.info("Recomputing predictions after unlearning.")
        self.F = np.full(len(y), self.init_score)
        for tree in self.trees:
            for idx, row in X.iterrows():
                self.F[idx] += self.learning_rate * tree.predict_single(row)

        logging.info("Unlearning process completed.")

def evaluate_model(model, X_test, y_test):
    preds = model.predict_proba(X_test)
    preds_binary = (preds >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds_binary)
    prec = precision_score(y_test, preds_binary, zero_division=0)
    rec = recall_score(y_test, preds_binary, zero_division=0)
    f1 = f1_score(y_test, preds_binary, zero_division=0)
    ll = log_loss(y_test, preds)
    return acc, prec, rec, f1, ll

def main(num_remove_percentage=5, output_csv='unlearning_results.csv'):
    # Step 1: Fetch datasets
    datasets = fetch_binary_classification_datasets_under_1000()
    logging.info(f"Fetched {len(datasets)} binary classification datasets.")

    if len(datasets) == 0:
        logging.error("No datasets found. Exiting.")
        return

    # Prepare a list to collect results
    results = []

    # For demonstration, we'll process only the first 5 datasets
    for did, name in tqdm(datasets, desc="Processing Datasets"):
        logging.info(f"\nProcessing Dataset ID: {did}, Name: {name}")
        loaded = load_and_prepare_dataset(did)
        if loaded is None:
            continue
        X, y, dataset_name = loaded

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Reset indices to ensure alignment
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # Initialize and train the model
        logging.info("Training the original model...")
        gbm = GradientBoostedTrees(n_estimators=10, learning_rate=0.1, max_depth=3, min_samples_split=10)
        gbm.fit(X_train, y_train)

        # Evaluate the model before unlearning
        acc, prec, rec, f1, ll = evaluate_model(gbm, X_test, y_test)
        logging.info(f"Original Model - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, Log Loss: {ll:.4f}")

        # Determine the number of samples to remove based on the percentage
        num_remove = max(1, int((num_remove_percentage / 100) * len(X_train)))
        # Select random indices to remove
        indices_to_remove = random.sample(range(len(X_train)), num_remove)
        logging.info(f"Removing {num_remove} samples ({num_remove_percentage}%) from training data.")

        # Perform unlearning
        gbm.unlearn(X_train, y_train, indices_to_remove)

        # Evaluate the model after unlearning
        acc_unlearn, prec_unlearn, rec_unlearn, f1_unlearn, ll_unlearn = evaluate_model(gbm, X_test, y_test)
        logging.info(f"Unlearned Model - Accuracy: {acc_unlearn:.4f}, Precision: {prec_unlearn:.4f}, Recall: {rec_unlearn:.4f}, F1: {f1_unlearn:.4f}, Log Loss: {ll_unlearn:.4f}")

        # Retrain a baseline model without the removed data
        logging.info("Retraining the baseline model without the removed samples...")
        X_train_retrained = X_train.drop(index=indices_to_remove).reset_index(drop=True)
        y_train_retrained = y_train.drop(index=indices_to_remove).reset_index(drop=True)
        baseline_model = GradientBoostedTrees(n_estimators=10, learning_rate=0.1, max_depth=3, min_samples_split=10)
        baseline_model.fit(X_train_retrained, y_train_retrained)

        # Evaluate the baseline model
        acc_base, prec_base, rec_base, f1_base, ll_base = evaluate_model(baseline_model, X_test, y_test)
        logging.info(f"Retrained Model - Accuracy: {acc_base:.4f}, Precision: {prec_base:.4f}, Recall: {rec_base:.4f}, F1: {f1_base:.4f}, Log Loss: {ll_base:.4f}")

        # Compare Unlearned Model with Retrained Model
        logging.info("Comparison between Unlearned Model and Retrained Model:")
        acc_diff = abs(acc_unlearn - acc_base)
        prec_diff = abs(prec_unlearn - prec_base)
        rec_diff = abs(rec_unlearn - rec_base)
        f1_diff = abs(f1_unlearn - f1_base)
        ll_diff = abs(ll_unlearn - ll_base)
        logging.info(f"Accuracy Difference: {acc_diff:.4f}")
        logging.info(f"Precision Difference: {prec_diff:.4f}")
        logging.info(f"Recall Difference: {rec_diff:.4f}")
        logging.info(f"F1 Score Difference: {f1_diff:.4f}")
        logging.info(f"Log Loss Difference: {ll_diff:.4f}")

        # Append results to the list
        results.append({
            'Dataset ID': did,
            'Dataset Name': name,
            'Original Accuracy': acc,
            'Original Precision': prec,
            'Original Recall': rec,
            'Original F1': f1,
            'Original Log Loss': ll,
            'Unlearned Accuracy': acc_unlearn,
            'Unlearned Precision': prec_unlearn,
            'Unlearned Recall': rec_unlearn,
            'Unlearned F1': f1_unlearn,
            'Unlearned Log Loss': ll_unlearn,
            'Retrained Accuracy': acc_base,
            'Retrained Precision': prec_base,
            'Retrained Recall': rec_base,
            'Retrained F1': f1_base,
            'Retrained Log Loss': ll_base,
            'Accuracy Difference': acc_diff,
            'Precision Difference': prec_diff,
            'Recall Difference': rec_diff,
            'F1 Difference': f1_diff,
            'Log Loss Difference': ll_diff,
            'Samples Removed': num_remove
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Define CSV file path
    csv_file = output_csv

    # Check if the file already exists
    if os.path.exists(csv_file):
        # If it exists, append without headers
        results_df.to_csv(csv_file, mode='a', index=False, header=False)
    else:
        # If it does not exist, write with headers
        results_df.to_csv(csv_file, mode='w', index=False)

    logging.info(f"\nAll results have been written to {csv_file}")

    # Optionally, display the results
    print("\nSummary of Results:")
    print(results_df)
    
    # Optional: Implement membership inference tests here

if __name__ == "__main__":
    # Parse command-line arguments for flexibility
    parser = argparse.ArgumentParser(description='Gradient Boosted Trees with Machine Unlearning')
    parser.add_argument('--remove_percentage', type=float, default=5.0,
                        help='Percentage of training samples to remove for unlearning (default: 5.0)')
    parser.add_argument('--output_csv', type=str, default='gbt_unlearning_results.csv',
                        help='Path to the output CSV file (default: gbt_unlearning_results.csv)')
    args = parser.parse_args()

    main(num_remove_percentage=args.remove_percentage, output_csv=args.output_csv)
