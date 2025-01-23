import random
import numpy as np
import openml
from sklearn.preprocessing import LabelEncoder
from fetch_datasets import fetch_datasets_less_than_1000_samples
from train_model import train_xgboost_with_metadata
from unlearn import unlearn_points, update_booster_with_metadata
from membership_inference import membership_inference_simple
from visualize_model import plot_first_tree, plot_feature_importance, plot_regression_predictions

def main():
    datasets_info = fetch_datasets_less_than_1000_samples()

    # For demonstration, let's only run on up to 3 datasets
    random.shuffle(datasets_info)
    datasets_info = datasets_info[:3]

    for did, name, is_classification in datasets_info:
        print(f"\n=== Working on dataset {did} ({name}), classification={is_classification} ===")

        # Load data
        try:
            X, y, is_class = load_openml_dataset(did)
            # We rely on 'is_class' from our load function to override the prior if needed
            is_classification = is_class

            # Train model
            booster, data_splits, metadata = train_xgboost_with_metadata(
                X, y, 
                is_classification=is_classification, 
                test_size=0.2, 
                num_rounds=10
            )
            X_train, X_test, y_train, y_test = data_splits

            # (A) Evaluate membership inference *before* unlearning
            # Let's pick some subset from X_train, say 10 points
            num_remove = min(10, len(X_train))
            X_remove = X_train[:num_remove]
            y_remove = y_train[:num_remove]

            # For membership inference, let's pick an equally sized random subset of X_test as "non-members"
            # or if test set < 10, we skip
            if len(X_test) < num_remove:
                print("Skipping membership inference due to small test set.")
            else:
                X_unknown = X_test[:num_remove]
                y_unknown = y_test[:num_remove]
                membership_guess_before, loss_known_before, loss_unknown_before = membership_inference_simple(
                    booster, X_remove, y_remove, X_unknown, y_unknown, is_classification
                )
                print("Membership inference BEFORE unlearning:")
                print(f" - Train Loss: {loss_known_before:.4f}, Unknown Loss: {loss_unknown_before:.4f}")
                print(f" - Membership guess: {membership_guess_before}")

            # (B) Perform unlearning
            updated_md = unlearn_points(
                booster,
                metadata,
                X_remove, 
                y_remove,
                is_classification=is_classification,
                lambda_reg=1.0
            )
            booster = update_booster_with_metadata(booster, updated_md)

            # (C) Evaluate membership inference *after* unlearning
            if len(X_test) >= num_remove:
                membership_guess_after, loss_known_after, loss_unknown_after = membership_inference_simple(
                    booster, X_remove, y_remove, X_unknown, y_unknown, is_classification
                )
                print("Membership inference AFTER unlearning:")
                print(f" - Train Loss: {loss_known_after:.4f}, Unknown Loss: {loss_unknown_after:.4f}")
                print(f" - Membership guess: {membership_guess_after}")

            # (D) Visualization
            # 1. Plot the first tree
            plot_first_tree(booster, filename=f"tree_plot_{did}.pdf")
            # 2. Plot feature importance
            plot_feature_importance(booster, filename=f"feature_importance_{did}.pdf")
            # 3. If regression, plot predicted vs. actual on test
            if not is_classification:
                plot_regression_predictions(booster, X_test, y_test, filename=f"regression_plot_{did}.pdf")

            print(f"Finished dataset {did} ({name}). Visuals saved.")

        except Exception as e:
            print(f"Error processing dataset {did} ({name}): {e}")

if __name__ == "__main__":
    main()
