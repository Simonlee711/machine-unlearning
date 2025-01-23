import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split

def get_xgb_params(is_classification, num_classes=None):
    """
    Returns a dictionary of XGBoost parameters depending on classification or regression.
    """
    if is_classification:
        if num_classes and num_classes > 2:
            params = {
                'objective': 'multi:softprob',
                'num_class': num_classes,
                'max_depth': 3,
                'learning_rate': 0.1,
                'lambda': 1,
                'verbosity': 0
            }
        else:
            params = {
                'objective': 'binary:logistic',
                'max_depth': 3,
                'learning_rate': 0.1,
                'lambda': 1,
                'verbosity': 0
            }
    else:
        # Regression
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 3,
            'learning_rate': 0.1,
            'lambda': 1,
            'verbosity': 0
        }
    return params

def train_xgboost_with_metadata(X, y, is_classification, test_size=0.2, num_rounds=10, random_state=42):
    """
    Splits the data into train/test, trains an XGBoost model, and extracts internal
    metadata (sum of gradients, sum of hessians, leaf value).
    Returns:
      booster, (X_train, X_test, y_train, y_test), metadata
    """

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if is_classification else None
    )

    # If classification, figure out # classes
    num_classes = None
    if is_classification:
        unique_labels = np.unique(y_train)
        num_classes = len(unique_labels)

    params = get_xgb_params(is_classification, num_classes)
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds
    )

    # Extract metadata: parse get_dump(with_stats=True)
    # Example structure:
    # 0:[f0<2] yes=1,no=2,missing=1,gain=...,cover=...
    #     1:leaf=...,cover=...,sum_grad=...,sum_hess=...
    metadata = []
    model_dump = booster.get_dump(with_stats=True)
    for tree in model_dump:
        tree_metadata = []
        leaf_index = 0
        for line in tree.splitlines():
            if 'leaf=' in line:
                # Attempt to parse grad/hess from line, if present
                # Typical line: "2:leaf=0.5,cover=5,sum_grad=1.5,sum_hess=3"
                segs = line.split(',')
                sum_grad, sum_hess, leaf_val = None, None, None
                for s in segs:
                    s = s.strip()
                    if s.startswith("leaf="):
                        leaf_val = float(s.split('=')[1])
                    if s.startswith("sum_grad="):
                        sum_grad = float(s.split('=')[1])
                    if s.startswith("sum_hess="):
                        sum_hess = float(s.split('=')[1])
                if sum_grad is not None and sum_hess is not None and leaf_val is not None:
                    tree_metadata.append({
                        'sum_grad': sum_grad,
                        'sum_hess': sum_hess,
                        'leaf_value': leaf_val
                    })
                else:
                    # Fallback if not found
                    tree_metadata.append({
                        'sum_grad': 0.0,
                        'sum_hess': 0.0,
                        'leaf_value': leaf_val if leaf_val is not None else 0.0
                    })
                leaf_index += 1
        metadata.append(tree_metadata)

    return booster, (X_train, X_test, y_train, y_test), metadata
