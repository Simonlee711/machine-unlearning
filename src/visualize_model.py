import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

def plot_first_tree(booster, filename="tree_plot.pdf"):
    """
    Plot the first tree of the booster to a file (PDF, PNG, etc.).
    Requires 'graphviz' installed and Python graphviz package.
    """
    ax = xgb.plot_tree(booster, num_trees=0)
    fig = ax.figure
    fig.savefig(filename)
    plt.close(fig)

def plot_feature_importance(booster, filename="feature_importance.pdf"):
    """
    Plot the gain-based feature importance and save to file.
    """
    xgb.plot_importance(booster)
    plt.savefig(filename)
    plt.close()

def plot_regression_predictions(booster, X_test, y_test, filename="regression_plot.pdf"):
    """
    For a regression problem, plot predicted vs. actual.
    """
    dtest = xgb.DMatrix(X_test)
    preds = booster.predict(dtest)
    plt.figure()
    plt.scatter(y_test, preds, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("XGBoost Predictions vs. Actual")
    # Plot a 45-degree line
    mn = min(min(y_test), min(preds))
    mx = max(max(y_test), max(preds))
    plt.plot([mn, mx], [mn, mx], color='red', linestyle='--')
    plt.savefig(filename)
    plt.close()
