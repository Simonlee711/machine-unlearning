import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ------------------------------------------------
# 1. Generate Synthetic Data
# ------------------------------------------------
np.random.seed(42)
N = 1000
X = np.random.rand(N, 5)  # 5 features
y = 5*X[:,0] - 3*X[:,1] + 2*X[:,2] + np.random.randn(N)*0.1  # a linear-ish target

# Convert to XGBoost DMatrix
dtrain_full = xgb.DMatrix(X, label=y)

# ------------------------------------------------
# 2. Train an XGBoost Model (on full data)
# ------------------------------------------------
# You can tune these params as needed.
params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'eta': 0.1,           # learning rate
    'eval_metric': 'rmse' # root mean squared error
}

num_boost_round = 50

print("Training original model on all data...")
model_full = xgb.train(params, dtrain_full, num_boost_round=num_boost_round)
pred_full = model_full.predict(dtrain_full)
rmse_full = np.sqrt(mean_squared_error(y, pred_full))
print(f"RMSE (original model on all data): {rmse_full:.4f}")

# ------------------------------------------------
# 3. Identify Samples to Remove (simulate 'revoked' data)
# ------------------------------------------------
# For example, let's remove 50 random points from the training set.
num_remove = 50
indices_all = np.arange(N)
np.random.shuffle(indices_all)
removed_indices = indices_all[:num_remove]
remaining_indices = indices_all[num_remove:]

print(f"Removing {num_remove} samples from the training set...")

# ------------------------------------------------
# 4. Method A: Full Retraining (Exact Unlearning)
# ------------------------------------------------
X_remaining = X[remaining_indices]
y_remaining = y[remaining_indices]

dtrain_remaining = xgb.DMatrix(X_remaining, label=y_remaining)

print("\n[Method A] Full retraining on remaining data...")
model_retrained = xgb.train(params, dtrain_remaining, num_boost_round=num_boost_round)

# Evaluate on the *same* remaining data
pred_retrained = model_retrained.predict(dtrain_remaining)
rmse_retrained = np.sqrt(mean_squared_error(y_remaining, pred_retrained))
print(f"RMSE (fully retrained model on remaining data): {rmse_retrained:.4f}")

# ------------------------------------------------
# 5. Method B: Partial Retraining (Approximate Unlearning)
# ------------------------------------------------
# We keep the first 'k' trees from the original model, and retrain the rest on the updated data.
# The assumption: The first 'k' trees won't drastically change if we remove a small fraction of data.

k = 30  # number of trees to keep from the original model
# That means we'll "freeze" the first 30 trees, and retrain the last (num_boost_round - k) trees.

# A. Extract raw text dump from the original model
dump_list_full = model_full.get_dump()  # list of strings, each string = 1 tree

# B. Keep the first k trees
frozen_trees = dump_list_full[:k]

# C. Create a new booster that starts with those k trees
#    We'll build it by converting the 'frozen_trees' to a text model, then load it.
temp_model_path = "temp_model.txt"
with open(temp_model_path, "w") as f:
    for tree_text in frozen_trees:
        f.write(tree_text + "\n")

booster_partial = xgb.Booster(params=params)
booster_partial.load_model(temp_model_path)

# D. Now we need to "continue training" for (num_boost_round - k) more rounds
#    on the updated data that excludes the removed points.
#    We'll do this by specifying 'process_type'='update' and 'updater'='refresh'
#    so we can keep the existing trees and only add new ones. 
#    Alternatively, we can treat it like an initial model but we want to *append* new trees.

num_new_trees = num_boost_round - k
# We'll do an internal loop to build new trees. 
# However, standard xgb.train(...) won't "append" trees to an existing booster automatically
# unless we specify the right parameters. We'll do it iteration by iteration.

dtrain_partial = xgb.DMatrix(X_remaining, label=y_remaining)

for _ in range(num_new_trees):
    booster_partial.update(dtrain_partial, iteration=0)  # iteration=0 is typical for each round

# Now we have a model with k "frozen" trees + newly trained (num_new_trees) trees
# But we need to check how many total trees are inside booster_partial now.
# By default, 'update()' with 'updater=refresh' modifies existing trees. 
# That is not necessarily what we want if we want to *add* new trees.

# ---
# In practice, partial "freezing" is tricky with the built-in Python API for XGBoost, 
# because 'updater=refresh' will refresh existing trees rather than add new ones.
# 
# Another approach is:
#    1) Train a new model with fewer boost rounds (k) on the original data, 
#       then load that model, 
#    2) Then continue training on the updated dataset for 'num_new_trees' more rounds.
# 
# Let’s demonstrate that simpler approach below:
# ---

# STEP B1: Train a separate model on the full data but with only k estimators
print(f"\n[Method B] Re-producing the partial freeze by separate training with k={k} trees on FULL data...")
model_first_k = xgb.train(params, dtrain_full, num_boost_round=k)

# STEP B2: Now continue training from that model for the last (num_boost_round - k) trees but on the REMAINING data
params_for_update = params.copy()
# By default, xgb.train does not continue from an existing model. 
# We can do:
model_partial = xgb.train(
    params_for_update,
    dtrain_remaining,
    num_boost_round=(num_boost_round - k),
    xgb_model=model_first_k  # start from model_first_k's booster
)

# Now 'model_partial' effectively has k trees trained on the full data + 
# (num_boost_round - k) trees trained only on the remaining data.

# Evaluate it:
pred_partial = model_partial.predict(dtrain_remaining)
rmse_partial = np.sqrt(mean_squared_error(y_remaining, pred_partial))
print(f"RMSE (partial retraining: keep first {k} trees, retrain last {num_boost_round - k} on remaining): {rmse_partial:.4f}")

# Compare with the fully retrained model:
print(f"RMSE difference vs fully retrained: {rmse_partial - rmse_retrained:.4f}")

# ------------------------------------------------
# 6. Compare Models
# ------------------------------------------------

print("\nSummary of results:")
print(f" - Original model (trained on ALL data) RMSE: {rmse_full:.4f}  (on its original training set)")
print(f" - Fully retrained model (Method A) RMSE on REMAINING data: {rmse_retrained:.4f}")
print(f" - Partial retraining model (Method B) RMSE on REMAINING data: {rmse_partial:.4f}")

print("\nNote: The partial retraining method is faster than full retraining (if k is large),")
print("but may still retain some influence from the removed data in the first k trees.")
print("Full retraining is the only guaranteed 'exact unlearning' method.")
