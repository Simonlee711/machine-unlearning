import numpy as np
from sklearn.metrics import log_loss, mean_squared_error

def membership_inference_simple(booster, X_known, y_known, X_unknown, y_unknown, is_classification):
    """
    Compute the average loss on X_known vs. X_unknown. If known-loss is significantly lower,
    an attacker might guess membership for certain records.
    Returns membership_scores for each sample in X_unknown (higher => more likely trained).
    """

    # Convert to DMatrix
    import xgboost as xgb
    dknown = xgb.DMatrix(X_known, label=y_known)
    dunknown = xgb.DMatrix(X_unknown, label=y_unknown)

    # Predictions
    preds_known = booster.predict(dknown)
    preds_unknown = booster.predict(dunknown)

    if is_classification:
        # If multi:softprob or binary:logistic, compute cross-entropy
        # Checking shape to see if multi-class
        if preds_known.ndim == 1:
            # binary:logistic
            loss_known = log_loss(y_known, preds_known, eps=1e-7)
            loss_unknown = log_loss(y_unknown, preds_unknown, eps=1e-7)
        else:
            # multi-class
            # y_known must be integer for scikit's log_loss with prob arrays
            loss_known = log_loss(y_known, preds_known, eps=1e-7)
            loss_unknown = log_loss(y_unknown, preds_unknown, eps=1e-7)

        # For each sample in X_unknown, compute the sample-level cross-entropy
        sample_level_loss_unknown = []
        if preds_unknown.ndim == 1:
            # binary
            for i in range(len(y_unknown)):
                # compute cross-entropy manually for each sample
                p = np.clip(preds_unknown[i], 1e-7, 1 - 1e-7)
                ll = -(y_unknown[i] * np.log(p) + (1 - y_unknown[i]) * np.log(1 - p))
                sample_level_loss_unknown.append(ll)
        else:
            # multi-class
            for i in range(len(y_unknown)):
                p_vec = np.clip(preds_unknown[i], 1e-7, 1.0)
                y_int = int(y_unknown[i])
                ll = -np.log(p_vec[y_int])
                sample_level_loss_unknown.append(ll)

        sample_level_loss_unknown = np.array(sample_level_loss_unknown)
        # Heuristic: if sample loss < some threshold => guess "member"
        # threshold could be the average training loss or an offset from it
        threshold = loss_known  # simple approach
        membership_guess = (sample_level_loss_unknown < threshold).astype(int)

    else:
        # Regression => use MSE as a membership check
        loss_known = mean_squared_error(y_known, preds_known)
        loss_unknown = mean_squared_error(y_unknown, preds_unknown)

        # Sample-level error
        sample_level_err_unknown = (preds_unknown - y_unknown)**2
        # threshold is the average training MSE
        threshold = loss_known
        membership_guess = (sample_level_err_unknown < threshold).astype(int)

    return membership_guess, loss_known, loss_unknown
