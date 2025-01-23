import xgboost as xgb
import numpy as np


def compute_grad_hess(booster, X_remove, y_remove, objective):
    dremove = xgb.DMatrix(X_remove, label=y_remove)
    raw_preds = booster.predict(dremove, output_margin=True)

    if objective == 'binary:logistic':
        # gradient = p - y
        # hessian = p * (1 - p)
        p = 1.0 / (1.0 + np.exp(-raw_preds))
        grad = p - y_remove
        hess = p * (1.0 - p)
    elif objective == 'reg:squarederror':
        grad = raw_preds - y_remove
        hess = np.ones_like(y_remove)
    else:
        raise NotImplementedError(f"Objective {objective} not yet implemented.")
    return grad, hess

def unlearn_points(booster, metadata, X_remove, y_remove, objective, lambda_reg=1.0):
    grad, hess = compute_grad_hess(booster, X_remove, y_remove, objective)
    grad_sum = np.sum(grad)
    hess_sum = np.sum(hess)

    updated_metadata = []
    for tree_md in metadata:
        new_tree_md = []
        for leaf_stats in tree_md:
            leaf_copy = leaf_stats.copy()
            leaf_copy['sum_grad'] -= grad_sum
            leaf_copy['sum_hess'] -= hess_sum
            denom = leaf_copy['sum_hess'] + lambda_reg
            if denom <= 1e-12:
                leaf_copy['leaf_value'] = 0.0
            else:
                leaf_copy['leaf_value'] = -leaf_copy['sum_grad'] / denom
            new_tree_md.append(leaf_copy)
        updated_metadata.append(new_tree_md)
    return updated_metadata

def update_booster_with_metadata(booster, updated_metadata):
    trees = booster.get_dump(with_stats=True)
    new_trees_dump = []

    for tree_idx, (tree, tree_meta) in enumerate(zip(trees, updated_metadata)):
        updated_tree_lines = []
        leaf_counter = 0
        for line in tree.splitlines():
            if 'leaf=' in line:
                old_val_str = line.split('leaf=')[1].split(',')[0]
                new_val = tree_meta[leaf_counter]['leaf_value']
                line = line.replace(f"leaf={old_val_str}", f"leaf={new_val}")
                leaf_counter += 1
            updated_tree_lines.append(line)
        new_trees_dump.append("\n".join(updated_tree_lines))

    booster.set_dump(new_trees_dump)
    return booster
